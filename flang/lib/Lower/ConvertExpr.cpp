//===-- ConvertExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertExpr.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/BuiltinModules.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ComponentPath.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertConstant.h"
#include "flang/Lower/ConvertProcedureDesignator.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/CustomIntrinsicCall.h"
#include "flang/Lower/DumpEvaluateExpr.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Ragged.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Runtime/support.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>

#define DEBUG_TYPE "flang-lower-expr"

using namespace Fortran::runtime;

//===----------------------------------------------------------------------===//
// The composition and structure of Fortran::evaluate::Expr is defined in
// the various header files in include/flang/Evaluate. You are referred
// there for more information on these data structures. Generally speaking,
// these data structures are a strongly typed family of abstract data types
// that, composed as trees, describe the syntax of Fortran expressions.
//
// This part of the bridge can traverse these tree structures and lower them
// to the correct FIR representation in SSA form.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> generateArrayCoordinate(
    "gen-array-coor",
    llvm::cl::desc("in lowering create ArrayCoorOp instead of CoordinateOp"),
    llvm::cl::init(false));

// The default attempts to balance a modest allocation size with expected user
// input to minimize bounds checks and reallocations during dynamic array
// construction. Some user codes may have very large array constructors for
// which the default can be increased.
static llvm::cl::opt<unsigned> clInitialBufferSize(
    "array-constructor-initial-buffer-size",
    llvm::cl::desc(
        "set the incremental array construction buffer size (default=32)"),
    llvm::cl::init(32u));

// Lower TRANSPOSE as an "elemental" function that swaps the array
// expression's iteration space, so that no runtime call is needed.
// This lowering may help get rid of unnecessary creation of temporary
// arrays. Note that the runtime TRANSPOSE implementation may be different
// from the "inline" FIR, e.g. it may diagnose out-of-memory conditions
// during the temporary allocation whereas the inline implementation
// relies on AllocMemOp that will silently return null in case
// there is not enough memory.
//
// If it is set to false, then TRANSPOSE will be lowered using
// a runtime call. If it is set to true, then the lowering is controlled
// by LoweringOptions::optimizeTranspose bit (see isTransposeOptEnabled
// function in this file).
static llvm::cl::opt<bool> optimizeTranspose(
    "opt-transpose",
    llvm::cl::desc("lower transpose without using a runtime call"),
    llvm::cl::init(true));

// When copy-in/copy-out is generated for a boxed object we may
// either produce loops to copy the data or call the Fortran runtime's
// Assign function. Since the data copy happens under a runtime check
// (for IsContiguous) the copy loops can hardly provide any value
// to optimizations, instead, the optimizer just wastes compilation
// time on these loops.
//
// This internal option will force the loops generation, when set
// to true. It is false by default.
//
// Note that for copy-in/copy-out of non-boxed objects (e.g. for passing
// arguments by value) we always generate loops. Since the memory for
// such objects is contiguous, it may be better to expose them
// to the optimizer.
static llvm::cl::opt<bool> inlineCopyInOutForBoxes(
    "inline-copyinout-for-boxes",
    llvm::cl::desc(
        "generate loops for copy-in/copy-out of objects with descriptors"),
    llvm::cl::init(false));

/// The various semantics of a program constituent (or a part thereof) as it may
/// appear in an expression.
///
/// Given the following Fortran declarations.
/// ```fortran
///   REAL :: v1, v2, v3
///   REAL, POINTER :: vp1
///   REAL :: a1(c), a2(c)
///   REAL ELEMENTAL FUNCTION f1(arg) ! array -> array
///   FUNCTION f2(arg)                ! array -> array
///   vp1 => v3       ! 1
///   v1 = v2 * vp1   ! 2
///   a1 = a1 + a2    ! 3
///   a1 = f1(a2)     ! 4
///   a1 = f2(a2)     ! 5
/// ```
///
/// In line 1, `vp1` is a BoxAddr to copy a box value into. The box value is
/// constructed from the DataAddr of `v3`.
/// In line 2, `v1` is a DataAddr to copy a value into. The value is constructed
/// from the DataValue of `v2` and `vp1`. DataValue is implicitly a double
/// dereference in the `vp1` case.
/// In line 3, `a1` and `a2` on the rhs are RefTransparent. The `a1` on the lhs
/// is CopyInCopyOut as `a1` is replaced elementally by the additions.
/// In line 4, `a2` can be RefTransparent, ByValueArg, RefOpaque, or BoxAddr if
/// `arg` is declared as C-like pass-by-value, VALUE, INTENT(?), or ALLOCATABLE/
/// POINTER, respectively. `a1` on the lhs is CopyInCopyOut.
///  In line 5, `a2` may be DataAddr or BoxAddr assuming f2 is transformational.
///  `a1` on the lhs is again CopyInCopyOut.
enum class ConstituentSemantics {
  // Scalar data reference semantics.
  //
  // For these let `v` be the location in memory of a variable with value `x`
  DataValue, // refers to the value `x`
  DataAddr,  // refers to the address `v`
  BoxValue,  // refers to a box value containing `v`
  BoxAddr,   // refers to the address of a box value containing `v`

  // Array data reference semantics.
  //
  // For these let `a` be the location in memory of a sequence of value `[xs]`.
  // Let `x_i` be the `i`-th value in the sequence `[xs]`.

  // Referentially transparent. Refers to the array's value, `[xs]`.
  RefTransparent,
  // Refers to an ephemeral address `tmp` containing value `x_i` (15.5.2.3.p7
  // note 2). (Passing a copy by reference to simulate pass-by-value.)
  ByValueArg,
  // Refers to the merge of array value `[xs]` with another array value `[ys]`.
  // This merged array value will be written into memory location `a`.
  CopyInCopyOut,
  // Similar to CopyInCopyOut but `a` may be a transient projection (rather than
  // a whole array).
  ProjectedCopyInCopyOut,
  // Similar to ProjectedCopyInCopyOut, except the merge value is not assigned
  // automatically by the framework. Instead, and address for `[xs]` is made
  // accessible so that custom assignments to `[xs]` can be implemented.
  CustomCopyInCopyOut,
  // Referentially opaque. Refers to the address of `x_i`.
  RefOpaque
};

/// Convert parser's INTEGER relational operators to MLIR.  TODO: using
/// unordered, but we may want to cons ordered in certain situation.
static mlir::arith::CmpIPredicate
translateRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::arith::CmpIPredicate::slt;
  case Fortran::common::RelationalOperator::LE:
    return mlir::arith::CmpIPredicate::sle;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::arith::CmpIPredicate::eq;
  case Fortran::common::RelationalOperator::NE:
    return mlir::arith::CmpIPredicate::ne;
  case Fortran::common::RelationalOperator::GT:
    return mlir::arith::CmpIPredicate::sgt;
  case Fortran::common::RelationalOperator::GE:
    return mlir::arith::CmpIPredicate::sge;
  }
  llvm_unreachable("unhandled INTEGER relational operator");
}

/// Convert parser's REAL relational operators to MLIR.
/// The choice of order (O prefix) vs unorder (U prefix) follows Fortran 2018
/// requirements in the IEEE context (table 17.1 of F2018). This choice is
/// also applied in other contexts because it is easier and in line with
/// other Fortran compilers.
/// FIXME: The signaling/quiet aspect of the table 17.1 requirement is not
/// fully enforced. FIR and LLVM `fcmp` instructions do not give any guarantee
/// whether the comparison will signal or not in case of quiet NaN argument.
static mlir::arith::CmpFPredicate
translateFloatRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::arith::CmpFPredicate::OLT;
  case Fortran::common::RelationalOperator::LE:
    return mlir::arith::CmpFPredicate::OLE;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::arith::CmpFPredicate::OEQ;
  case Fortran::common::RelationalOperator::NE:
    return mlir::arith::CmpFPredicate::UNE;
  case Fortran::common::RelationalOperator::GT:
    return mlir::arith::CmpFPredicate::OGT;
  case Fortran::common::RelationalOperator::GE:
    return mlir::arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("unhandled REAL relational operator");
}

static mlir::Value genActualIsPresentTest(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          fir::ExtendedValue actual) {
  if (const auto *ptrOrAlloc = actual.getBoxOf<fir::MutableBoxValue>())
    return fir::factory::genIsAllocatedOrAssociatedTest(builder, loc,
                                                        *ptrOrAlloc);
  // Optional case (not that optional allocatable/pointer cannot be absent
  // when passed to CMPLX as per 15.5.2.12 point 3 (7) and (8)). It is
  // therefore possible to catch them in the `then` case above.
  return builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                          fir::getBase(actual));
}

/// Convert the array_load, `load`, to an extended value. If `path` is not
/// empty, then traverse through the components designated. The base value is
/// `newBase`. This does not accept an array_load with a slice operand.
static fir::ExtendedValue
arrayLoadExtValue(fir::FirOpBuilder &builder, mlir::Location loc,
                  fir::ArrayLoadOp load, llvm::ArrayRef<mlir::Value> path,
                  mlir::Value newBase, mlir::Value newLen = {}) {
  // Recover the extended value from the load.
  if (load.getSlice())
    fir::emitFatalError(loc, "array_load with slice is not allowed");
  mlir::Type arrTy = load.getType();
  if (!path.empty()) {
    mlir::Type ty = fir::applyPathToType(arrTy, path);
    if (!ty)
      fir::emitFatalError(loc, "path does not apply to type");
    if (!ty.isa<fir::SequenceType>()) {
      if (fir::isa_char(ty)) {
        mlir::Value len = newLen;
        if (!len)
          len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
              load.getMemref());
        if (!len) {
          assert(load.getTypeparams().size() == 1 &&
                 "length must be in array_load");
          len = load.getTypeparams()[0];
        }
        return fir::CharBoxValue{newBase, len};
      }
      return newBase;
    }
    arrTy = ty.cast<fir::SequenceType>();
  }

  auto arrayToExtendedValue =
      [&](const llvm::SmallVector<mlir::Value> &extents,
          const llvm::SmallVector<mlir::Value> &origins) -> fir::ExtendedValue {
    mlir::Type eleTy = fir::unwrapSequenceType(arrTy);
    if (fir::isa_char(eleTy)) {
      mlir::Value len = newLen;
      if (!len)
        len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
            load.getMemref());
      if (!len) {
        assert(load.getTypeparams().size() == 1 &&
               "length must be in array_load");
        len = load.getTypeparams()[0];
      }
      return fir::CharArrayBoxValue(newBase, len, extents, origins);
    }
    return fir::ArrayBoxValue(newBase, extents, origins);
  };
  // Use the shape op, if there is one.
  mlir::Value shapeVal = load.getShape();
  if (shapeVal) {
    if (!mlir::isa<fir::ShiftOp>(shapeVal.getDefiningOp())) {
      auto extents = fir::factory::getExtents(shapeVal);
      auto origins = fir::factory::getOrigins(shapeVal);
      return arrayToExtendedValue(extents, origins);
    }
    if (!fir::isa_box_type(load.getMemref().getType()))
      fir::emitFatalError(loc, "shift op is invalid in this context");
  }

  // If we're dealing with the array_load op (not a subobject) and the load does
  // not have any type parameters, then read the extents from the original box.
  // The origin may be either from the box or a shift operation. Create and
  // return the array extended value.
  if (path.empty() && load.getTypeparams().empty()) {
    auto oldBox = load.getMemref();
    fir::ExtendedValue exv = fir::factory::readBoxValue(builder, loc, oldBox);
    auto extents = fir::factory::getExtents(loc, builder, exv);
    auto origins = fir::factory::getNonDefaultLowerBounds(builder, loc, exv);
    if (shapeVal) {
      // shapeVal is a ShiftOp and load.memref() is a boxed value.
      newBase = builder.create<fir::ReboxOp>(loc, oldBox.getType(), oldBox,
                                             shapeVal, /*slice=*/mlir::Value{});
      origins = fir::factory::getOrigins(shapeVal);
    }
    return fir::substBase(arrayToExtendedValue(extents, origins), newBase);
  }
  TODO(loc, "path to a POINTER, ALLOCATABLE, or other component that requires "
            "dereferencing; generating the type parameters is a hard "
            "requirement for correctness.");
}

/// Place \p exv in memory if it is not already a memory reference. If
/// \p forceValueType is provided, the value is first casted to the provided
/// type before being stored (this is mainly intended for logicals whose value
/// may be `i1` but needed to be stored as Fortran logicals).
static fir::ExtendedValue
placeScalarValueInMemory(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::ExtendedValue &exv,
                         mlir::Type storageType) {
  mlir::Value valBase = fir::getBase(exv);
  if (fir::conformsWithPassByRef(valBase.getType()))
    return exv;

  assert(!fir::hasDynamicSize(storageType) &&
         "only expect statically sized scalars to be by value");

  // Since `a` is not itself a valid referent, determine its value and
  // create a temporary location at the beginning of the function for
  // referencing.
  mlir::Value val = builder.createConvert(loc, storageType, valBase);
  mlir::Value temp = builder.createTemporary(
      loc, storageType,
      llvm::ArrayRef<mlir::NamedAttribute>{
          Fortran::lower::getAdaptToByRefAttr(builder)});
  builder.create<fir::StoreOp>(loc, val, temp);
  return fir::substBase(exv, temp);
}

// Copy a copy of scalar \p exv in a new temporary.
static fir::ExtendedValue
createInMemoryScalarCopy(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::ExtendedValue &exv) {
  assert(exv.rank() == 0 && "input to scalar memory copy must be a scalar");
  if (exv.getCharBox() != nullptr)
    return fir::factory::CharacterExprHelper{builder, loc}.createTempFrom(exv);
  if (fir::isDerivedWithLenParameters(exv))
    TODO(loc, "copy derived type with length parameters");
  mlir::Type type = fir::unwrapPassByRefType(fir::getBase(exv).getType());
  fir::ExtendedValue temp = builder.createTemporary(loc, type);
  fir::factory::genScalarAssignment(builder, loc, temp, exv);
  return temp;
}

// An expression with non-zero rank is an array expression.
template <typename A>
static bool isArray(const A &x) {
  return x.Rank() != 0;
}

/// Is this a variable wrapped in parentheses?
template <typename A>
static bool isParenthesizedVariable(const A &) {
  return false;
}
template <typename T>
static bool isParenthesizedVariable(const Fortran::evaluate::Expr<T> &expr) {
  using ExprVariant = decltype(Fortran::evaluate::Expr<T>::u);
  using Parentheses = Fortran::evaluate::Parentheses<T>;
  if constexpr (Fortran::common::HasMember<Parentheses, ExprVariant>) {
    if (const auto *parentheses = std::get_if<Parentheses>(&expr.u))
      return Fortran::evaluate::IsVariable(parentheses->left());
    return false;
  } else {
    return std::visit([&](const auto &x) { return isParenthesizedVariable(x); },
                      expr.u);
  }
}

/// Generate a load of a value from an address. Beware that this will lose
/// any dynamic type information for polymorphic entities (note that unlimited
/// polymorphic cannot be loaded and must not be provided here).
static fir::ExtendedValue genLoad(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  const fir::ExtendedValue &addr) {
  return addr.match(
      [](const fir::CharBoxValue &box) -> fir::ExtendedValue { return box; },
      [&](const fir::PolymorphicValue &p) -> fir::ExtendedValue {
        if (fir::unwrapRefType(fir::getBase(p).getType())
                .isa<fir::RecordType>())
          return p;
        mlir::Value load = builder.create<fir::LoadOp>(loc, fir::getBase(p));
        return fir::PolymorphicValue(load, p.getSourceBox());
      },
      [&](const fir::UnboxedValue &v) -> fir::ExtendedValue {
        if (fir::unwrapRefType(fir::getBase(v).getType())
                .isa<fir::RecordType>())
          return v;
        return builder.create<fir::LoadOp>(loc, fir::getBase(v));
      },
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        return genLoad(builder, loc,
                       fir::factory::genMutableBoxRead(builder, loc, box));
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        return genLoad(builder, loc,
                       fir::factory::readBoxValue(builder, loc, box));
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(
            loc, "attempting to load whole array or procedure address");
      });
}

/// Create an optional dummy argument value from entity \p exv that may be
/// absent. This can only be called with numerical or logical scalar \p exv.
/// If \p exv is considered absent according to 15.5.2.12 point 1., the returned
/// value is zero (or false), otherwise it is the value of \p exv.
static fir::ExtendedValue genOptionalValue(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           const fir::ExtendedValue &exv,
                                           mlir::Value isPresent) {
  mlir::Type eleType = fir::getBaseTypeOf(exv);
  assert(exv.rank() == 0 && fir::isa_trivial(eleType) &&
         "must be a numerical or logical scalar");
  return builder
      .genIfOp(loc, {eleType}, isPresent,
               /*withElseRegion=*/true)
      .genThen([&]() {
        mlir::Value val = fir::getBase(genLoad(builder, loc, exv));
        builder.create<fir::ResultOp>(loc, val);
      })
      .genElse([&]() {
        mlir::Value zero = fir::factory::createZeroValue(builder, loc, eleType);
        builder.create<fir::ResultOp>(loc, zero);
      })
      .getResults()[0];
}

/// Create an optional dummy argument address from entity \p exv that may be
/// absent. If \p exv is considered absent according to 15.5.2.12 point 1., the
/// returned value is a null pointer, otherwise it is the address of \p exv.
static fir::ExtendedValue genOptionalAddr(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          const fir::ExtendedValue &exv,
                                          mlir::Value isPresent) {
  // If it is an exv pointer/allocatable, then it cannot be absent
  // because it is passed to a non-pointer/non-allocatable.
  if (const auto *box = exv.getBoxOf<fir::MutableBoxValue>())
    return fir::factory::genMutableBoxRead(builder, loc, *box);
  // If this is not a POINTER or ALLOCATABLE, then it is already an OPTIONAL
  // address and can be passed directly.
  return exv;
}

/// Create an optional dummy argument address from entity \p exv that may be
/// absent. If \p exv is considered absent according to 15.5.2.12 point 1., the
/// returned value is an absent fir.box, otherwise it is a fir.box describing \p
/// exv.
static fir::ExtendedValue genOptionalBox(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         const fir::ExtendedValue &exv,
                                         mlir::Value isPresent) {
  // Non allocatable/pointer optional box -> simply forward
  if (exv.getBoxOf<fir::BoxValue>())
    return exv;

  fir::ExtendedValue newExv = exv;
  // Optional allocatable/pointer -> Cannot be absent, but need to translate
  // unallocated/diassociated into absent fir.box.
  if (const auto *box = exv.getBoxOf<fir::MutableBoxValue>())
    newExv = fir::factory::genMutableBoxRead(builder, loc, *box);

  // createBox will not do create any invalid memory dereferences if exv is
  // absent. The created fir.box will not be usable, but the SelectOp below
  // ensures it won't be.
  mlir::Value box = builder.createBox(loc, newExv);
  mlir::Type boxType = box.getType();
  auto absent = builder.create<fir::AbsentOp>(loc, boxType);
  auto boxOrAbsent = builder.create<mlir::arith::SelectOp>(
      loc, boxType, isPresent, box, absent);
  return fir::BoxValue(boxOrAbsent);
}

/// Is this a call to an elemental procedure with at least one array argument?
static bool
isElementalProcWithArrayArgs(const Fortran::evaluate::ProcedureRef &procRef) {
  if (procRef.IsElemental())
    for (const std::optional<Fortran::evaluate::ActualArgument> &arg :
         procRef.arguments())
      if (arg && arg->Rank() != 0)
        return true;
  return false;
}
template <typename T>
static bool isElementalProcWithArrayArgs(const Fortran::evaluate::Expr<T> &) {
  return false;
}
template <>
bool isElementalProcWithArrayArgs(const Fortran::lower::SomeExpr &x) {
  if (const auto *procRef = std::get_if<Fortran::evaluate::ProcedureRef>(&x.u))
    return isElementalProcWithArrayArgs(*procRef);
  return false;
}

/// \p argTy must be a tuple (pair) of boxproc and integral types. Convert the
/// \p funcAddr argument to a boxproc value, with the host-association as
/// required. Call the factory function to finish creating the tuple value.
static mlir::Value
createBoxProcCharTuple(Fortran::lower::AbstractConverter &converter,
                       mlir::Type argTy, mlir::Value funcAddr,
                       mlir::Value charLen) {
  auto boxTy =
      argTy.cast<mlir::TupleType>().getType(0).cast<fir::BoxProcType>();
  mlir::Location loc = converter.getCurrentLocation();
  auto &builder = converter.getFirOpBuilder();

  // While character procedure arguments are expected here, Fortran allows
  // actual arguments of other types to be passed instead.
  // To support this, we cast any reference to the expected type or extract
  // procedures from their boxes if needed.
  mlir::Type fromTy = funcAddr.getType();
  mlir::Type toTy = boxTy.getEleTy();
  if (fir::isa_ref_type(fromTy))
    funcAddr = builder.createConvert(loc, toTy, funcAddr);
  else if (fromTy.isa<fir::BoxProcType>())
    funcAddr = builder.create<fir::BoxAddrOp>(loc, toTy, funcAddr);

  auto boxProc = [&]() -> mlir::Value {
    if (auto host = Fortran::lower::argumentHostAssocs(converter, funcAddr))
      return builder.create<fir::EmboxProcOp>(
          loc, boxTy, llvm::ArrayRef<mlir::Value>{funcAddr, host});
    return builder.create<fir::EmboxProcOp>(loc, boxTy, funcAddr);
  }();
  return fir::factory::createCharacterProcedureTuple(builder, loc, argTy,
                                                     boxProc, charLen);
}

/// Given an optional fir.box, returns an fir.box that is the original one if
/// it is present and it otherwise an unallocated box.
/// Absent fir.box are implemented as a null pointer descriptor. Generated
/// code may need to unconditionally read a fir.box that can be absent.
/// This helper allows creating a fir.box that can be read in all cases
/// outside of a fir.if (isPresent) region. However, the usages of the value
/// read from such box should still only be done in a fir.if(isPresent).
static fir::ExtendedValue
absentBoxToUnallocatedBox(fir::FirOpBuilder &builder, mlir::Location loc,
                          const fir::ExtendedValue &exv,
                          mlir::Value isPresent) {
  mlir::Value box = fir::getBase(exv);
  mlir::Type boxType = box.getType();
  assert(boxType.isa<fir::BoxType>() && "argument must be a fir.box");
  mlir::Value emptyBox =
      fir::factory::createUnallocatedBox(builder, loc, boxType, std::nullopt);
  auto safeToReadBox =
      builder.create<mlir::arith::SelectOp>(loc, isPresent, box, emptyBox);
  return fir::substBase(exv, safeToReadBox);
}

// Helper to get the ultimate first symbol. This works around the fact that
// symbol resolution in the front end doesn't always resolve a symbol to its
// ultimate symbol but may leave placeholder indirections for use and host
// associations.
template <typename A>
const Fortran::semantics::Symbol &getFirstSym(const A &obj) {
  return obj.GetFirstSymbol().GetUltimate();
}

// Helper to get the ultimate last symbol.
template <typename A>
const Fortran::semantics::Symbol &getLastSym(const A &obj) {
  return obj.GetLastSymbol().GetUltimate();
}

// Return true if TRANSPOSE should be lowered without a runtime call.
static bool
isTransposeOptEnabled(const Fortran::lower::AbstractConverter &converter) {
  return optimizeTranspose &&
         converter.getLoweringOptions().getOptimizeTranspose();
}

// A set of visitors to detect if the given expression
// is a TRANSPOSE call that should be lowered without using
// runtime TRANSPOSE implementation.
template <typename T>
static bool isOptimizableTranspose(const T &,
                                   const Fortran::lower::AbstractConverter &) {
  return false;
}

static bool
isOptimizableTranspose(const Fortran::evaluate::ProcedureRef &procRef,
                       const Fortran::lower::AbstractConverter &converter) {
  const Fortran::evaluate::SpecificIntrinsic *intrin =
      procRef.proc().GetSpecificIntrinsic();
  if (isTransposeOptEnabled(converter) && intrin &&
      intrin->name == "transpose") {
    const std::optional<Fortran::evaluate::ActualArgument> matrix =
        procRef.arguments().at(0);
    return !(matrix && matrix->GetType() && matrix->GetType()->IsPolymorphic());
  }
  return false;
}

template <typename T>
static bool
isOptimizableTranspose(const Fortran::evaluate::FunctionRef<T> &funcRef,
                       const Fortran::lower::AbstractConverter &converter) {
  return isOptimizableTranspose(
      static_cast<const Fortran::evaluate::ProcedureRef &>(funcRef), converter);
}

template <typename T>
static bool
isOptimizableTranspose(Fortran::evaluate::Expr<T> expr,
                       const Fortran::lower::AbstractConverter &converter) {
  // If optimizeTranspose is not enabled, return false right away.
  if (!isTransposeOptEnabled(converter))
    return false;

  return std::visit(
      [&](const auto &e) { return isOptimizableTranspose(e, converter); },
      expr.u);
}

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ScalarExprLowering {
public:
  using ExtValue = fir::ExtendedValue;

  explicit ScalarExprLowering(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              Fortran::lower::SymMap &symMap,
                              Fortran::lower::StatementContext &stmtCtx,
                              bool inInitializer = false)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap},
        inInitializer{inInitializer} {}

  ExtValue genExtAddr(const Fortran::lower::SomeExpr &expr) {
    return gen(expr);
  }

  /// Lower `expr` to be passed as a fir.box argument. Do not create a temp
  /// for the expr if it is a variable that can be described as a fir.box.
  ExtValue genBoxArg(const Fortran::lower::SomeExpr &expr) {
    bool saveUseBoxArg = useBoxArg;
    useBoxArg = true;
    ExtValue result = gen(expr);
    useBoxArg = saveUseBoxArg;
    return result;
  }

  ExtValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  /// Lower an expression that is a pointer or an allocatable to a
  /// MutableBoxValue.
  fir::MutableBoxValue
  genMutableBoxValue(const Fortran::lower::SomeExpr &expr) {
    // Pointers and allocatables can only be:
    //    - a simple designator "x"
    //    - a component designator "a%b(i,j)%x"
    //    - a function reference "foo()"
    //    - result of NULL() or NULL(MOLD) intrinsic.
    //    NULL() requires some context to be lowered, so it is not handled
    //    here and must be lowered according to the context where it appears.
    ExtValue exv = std::visit(
        [&](const auto &x) { return genMutableBoxValueImpl(x); }, expr.u);
    const fir::MutableBoxValue *mutableBox =
        exv.getBoxOf<fir::MutableBoxValue>();
    if (!mutableBox)
      fir::emitFatalError(getLoc(), "expr was not lowered to MutableBoxValue");
    return *mutableBox;
  }

  template <typename T>
  ExtValue genMutableBoxValueImpl(const T &) {
    // NULL() case should not be handled here.
    fir::emitFatalError(getLoc(), "NULL() must be lowered in its context");
  }

  /// A `NULL()` in a position where a mutable box is expected has the same
  /// semantics as an absent optional box value. Note: this code should
  /// be depreciated because the rank information is not known here. A
  /// scalar fir.box is created: it should not be cast to an array box type
  /// later, but there is no way to enforce that here.
  ExtValue genMutableBoxValueImpl(const Fortran::evaluate::NullPointer &) {
    mlir::Location loc = getLoc();
    mlir::Type noneTy = mlir::NoneType::get(builder.getContext());
    mlir::Type polyRefTy = fir::PointerType::get(noneTy);
    mlir::Type boxType = fir::BoxType::get(polyRefTy);
    mlir::Value nullConst = builder.createNullConstant(loc, polyRefTy);
    mlir::Value tempBox =
        builder.createTemporary(loc, boxType, /*shape=*/mlir::ValueRange{});
    mlir::Value nullBox = builder.create<fir::EmboxOp>(loc, boxType, nullConst);
    builder.create<fir::StoreOp>(loc, nullBox, tempBox);
    auto nullBoxTy = builder.getRefType(fir::BoxType::get(noneTy));
    return fir::MutableBoxValue(tempBox,
                                /*lenParameters=*/mlir::ValueRange{},
                                /*mutableProperties=*/{});
  }

  template <typename T>
  ExtValue
  genMutableBoxValueImpl(const Fortran::evaluate::FunctionRef<T> &funRef) {
    return genRawProcedureRef(funRef, converter.genType(toEvExpr(funRef)));
  }

  template <typename T>
  ExtValue
  genMutableBoxValueImpl(const Fortran::evaluate::Designator<T> &designator) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::SymbolRef &sym) -> ExtValue {
              return converter.getSymbolExtendedValue(*sym, &symMap);
            },
            [&](const Fortran::evaluate::Component &comp) -> ExtValue {
              return genComponent(comp);
            },
            [&](const auto &) -> ExtValue {
              fir::emitFatalError(getLoc(),
                                  "not an allocatable or pointer designator");
            }},
        designator.u);
  }

  template <typename T>
  ExtValue genMutableBoxValueImpl(const Fortran::evaluate::Expr<T> &expr) {
    return std::visit([&](const auto &x) { return genMutableBoxValueImpl(x); },
                      expr.u);
  }

  mlir::Location getLoc() { return location; }

  template <typename A>
  mlir::Value genunbox(const A &expr) {
    ExtValue e = genval(expr);
    if (const fir::UnboxedValue *r = e.getUnboxed())
      return *r;
    fir::emitFatalError(getLoc(), "unboxed expression expected");
  }

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    mlir::Type type =
        converter.genType(Fortran::common::TypeCategory::Integer, KIND);
    return builder.createIntegerConstant(getLoc(), type, value);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(bool value) {
    return builder.createBool(getLoc(), value);
  }

  mlir::Type getSomeKindInteger() { return builder.getIndexType(); }

  mlir::func::FuncOp getFunction(llvm::StringRef name,
                                 mlir::FunctionType funTy) {
    if (mlir::func::FuncOp func = builder.getNamedFunction(name))
      return func;
    return builder.createFunction(getLoc(), name, funTy);
  }

  template <typename OpTy>
  mlir::Value createCompareOp(mlir::arith::CmpIPredicate pred,
                              const ExtValue &left, const ExtValue &right) {
    if (const fir::UnboxedValue *lhs = left.getUnboxed())
      if (const fir::UnboxedValue *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::arith::CmpIPredicate pred) {
    ExtValue left = genval(ex.left());
    return createCompareOp<OpTy>(pred, left, genval(ex.right()));
  }

  template <typename OpTy>
  mlir::Value createFltCmpOp(mlir::arith::CmpFPredicate pred,
                             const ExtValue &left, const ExtValue &right) {
    if (const fir::UnboxedValue *lhs = left.getUnboxed())
      if (const fir::UnboxedValue *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::arith::CmpFPredicate pred) {
    ExtValue left = genval(ex.left());
    return createFltCmpOp<OpTy>(pred, left, genval(ex.right()));
  }

  /// Create a call to the runtime to compare two CHARACTER values.
  /// Precondition: This assumes that the two values have `fir.boxchar` type.
  mlir::Value createCharCompare(mlir::arith::CmpIPredicate pred,
                                const ExtValue &left, const ExtValue &right) {
    return fir::runtime::genCharCompare(builder, getLoc(), pred, left, right);
  }

  template <typename A>
  mlir::Value createCharCompare(const A &ex, mlir::arith::CmpIPredicate pred) {
    ExtValue left = genval(ex.left());
    return createCharCompare(pred, left, genval(ex.right()));
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  ExtValue gen(Fortran::semantics::SymbolRef sym) {
    fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym, &symMap);
    if (const auto *box = exv.getBoxOf<fir::MutableBoxValue>())
      return fir::factory::genMutableBoxRead(builder, getLoc(), *box);
    return exv;
  }

  ExtValue genLoad(const ExtValue &exv) {
    return ::genLoad(builder, getLoc(), exv);
  }

  ExtValue genval(Fortran::semantics::SymbolRef sym) {
    mlir::Location loc = getLoc();
    ExtValue var = gen(sym);
    if (const fir::UnboxedValue *s = var.getUnboxed())
      if (fir::isa_ref_type(s->getType())) {
        // A function with multiple entry points returning different types
        // tags all result variables with one of the largest types to allow
        // them to share the same storage.  A reference to a result variable
        // of one of the other types requires conversion to the actual type.
        fir::UnboxedValue addr = *s;
        if (Fortran::semantics::IsFunctionResult(sym)) {
          mlir::Type resultType = converter.genType(*sym);
          if (addr.getType() != resultType)
            addr = builder.createConvert(loc, builder.getRefType(resultType),
                                         addr);
        }
        return genLoad(addr);
      }
    return var;
  }

  ExtValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO(getLoc(), "BOZ");
  }

  /// Return indirection to function designated in ProcedureDesignator.
  /// The type of the function indirection is not guaranteed to match the one
  /// of the ProcedureDesignator due to Fortran implicit typing rules.
  ExtValue genval(const Fortran::evaluate::ProcedureDesignator &proc) {
    return Fortran::lower::convertProcedureDesignator(getLoc(), converter, proc,
                                                      symMap, stmtCtx);
  }
  ExtValue genval(const Fortran::evaluate::NullPointer &) {
    return builder.createNullConstant(getLoc());
  }

  static bool
  isDerivedTypeWithLenParameters(const Fortran::semantics::Symbol &sym) {
    if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType())
      if (const Fortran::semantics::DerivedTypeSpec *derived =
              declTy->AsDerived())
        return Fortran::semantics::CountLenParameters(*derived) > 0;
    return false;
  }

  /// A structure constructor is lowered two ways. In an initializer context,
  /// the entire structure must be constant, so the aggregate value is
  /// constructed inline. This allows it to be the body of a GlobalOp.
  /// Otherwise, the structure constructor is in an expression. In that case, a
  /// temporary object is constructed in the stack frame of the procedure.
  ExtValue genval(const Fortran::evaluate::StructureConstructor &ctor) {
    mlir::Location loc = getLoc();
    if (inInitializer)
      return Fortran::lower::genInlinedStructureCtorLit(converter, loc, ctor);
    mlir::Type ty = translateSomeExprToFIRType(converter, toEvExpr(ctor));
    auto recTy = ty.cast<fir::RecordType>();
    auto fieldTy = fir::FieldType::get(ty.getContext());
    mlir::Value res = builder.createTemporary(loc, recTy);
    mlir::Value box = builder.createBox(loc, fir::ExtendedValue{res});
    fir::runtime::genDerivedTypeInitialize(builder, loc, box);

    for (const auto &value : ctor.values()) {
      const Fortran::semantics::Symbol &sym = *value.first;
      const Fortran::lower::SomeExpr &expr = value.second.value();
      if (sym.test(Fortran::semantics::Symbol::Flag::ParentComp)) {
        ExtValue from = gen(expr);
        mlir::Type fromTy = fir::unwrapPassByRefType(
            fir::unwrapRefType(fir::getBase(from).getType()));
        mlir::Value resCast =
            builder.createConvert(loc, builder.getRefType(fromTy), res);
        fir::factory::genRecordAssignment(builder, loc, resCast, from);
        continue;
      }

      if (isDerivedTypeWithLenParameters(sym))
        TODO(loc, "component with length parameters in structure constructor");

      llvm::StringRef name = toStringRef(sym.name());
      // FIXME: type parameters must come from the derived-type-spec
      mlir::Value field = builder.create<fir::FieldIndexOp>(
          loc, fieldTy, name, ty,
          /*typeParams=*/mlir::ValueRange{} /*TODO*/);
      mlir::Type coorTy = builder.getRefType(recTy.getType(name));
      auto coor = builder.create<fir::CoordinateOp>(loc, coorTy,
                                                    fir::getBase(res), field);
      ExtValue to = fir::factory::componentToExtendedValue(builder, loc, coor);
      to.match(
          [&](const fir::UnboxedValue &toPtr) {
            ExtValue value = genval(expr);
            fir::factory::genScalarAssignment(builder, loc, to, value);
          },
          [&](const fir::CharBoxValue &) {
            ExtValue value = genval(expr);
            fir::factory::genScalarAssignment(builder, loc, to, value);
          },
          [&](const fir::ArrayBoxValue &) {
            Fortran::lower::createSomeArrayAssignment(converter, to, expr,
                                                      symMap, stmtCtx);
          },
          [&](const fir::CharArrayBoxValue &) {
            Fortran::lower::createSomeArrayAssignment(converter, to, expr,
                                                      symMap, stmtCtx);
          },
          [&](const fir::BoxValue &toBox) {
            fir::emitFatalError(loc, "derived type components must not be "
                                     "represented by fir::BoxValue");
          },
          [&](const fir::PolymorphicValue &) {
            TODO(loc, "polymorphic component in derived type assignment");
          },
          [&](const fir::MutableBoxValue &toBox) {
            if (toBox.isPointer()) {
              Fortran::lower::associateMutableBox(converter, loc, toBox, expr,
                                                  /*lbounds=*/std::nullopt,
                                                  stmtCtx);
              return;
            }
            // For allocatable components, a deep copy is needed.
            TODO(loc, "allocatable components in derived type assignment");
          },
          [&](const fir::ProcBoxValue &toBox) {
            TODO(loc, "procedure pointer component in derived type assignment");
          });
    }
    return res;
  }

  /// Lowering of an <i>ac-do-variable</i>, which is not a Symbol.
  ExtValue genval(const Fortran::evaluate::ImpliedDoIndex &var) {
    return converter.impliedDoBinding(toStringRef(var.name));
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    ExtValue exv = desc.base().IsSymbol() ? gen(getLastSym(desc.base()))
                                          : gen(desc.base().GetComponent());
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Location loc = getLoc();
    auto castResult = [&](mlir::Value v) {
      using ResTy = Fortran::evaluate::DescriptorInquiry::Result;
      return builder.createConvert(
          loc, converter.genType(ResTy::category, ResTy::kind), v);
    };
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      return castResult(fir::factory::readCharLen(builder, loc, exv));
    case Fortran::evaluate::DescriptorInquiry::Field::LowerBound:
      return castResult(fir::factory::readLowerBound(
          builder, loc, exv, desc.dimension(),
          builder.createIntegerConstant(loc, idxTy, 1)));
    case Fortran::evaluate::DescriptorInquiry::Field::Extent:
      return castResult(
          fir::factory::readExtent(builder, loc, exv, desc.dimension()));
    case Fortran::evaluate::DescriptorInquiry::Field::Rank:
      TODO(loc, "rank inquiry on assumed rank");
    case Fortran::evaluate::DescriptorInquiry::Field::Stride:
      // So far the front end does not generate this inquiry.
      TODO(loc, "stride inquiry");
    }
    llvm_unreachable("unknown descriptor inquiry");
  }

  ExtValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "type parameter inquiry");
  }

  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return fir::factory::Complex{builder, getLoc()}.extractComplexPart(
        cplx, isImagPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    return extractComplexPart(genunbox(part.left()), part.isImaginaryPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    mlir::Value input = genunbox(op.left());
    // Like LLVM, integer negation is the binary op "0 - value"
    mlir::Value zero = genIntegerConstant<KIND>(builder.getContext(), 0);
    return builder.create<mlir::arith::SubIOp>(getLoc(), zero, input);
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return builder.create<mlir::arith::NegFOp>(getLoc(), genunbox(op.left()));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return builder.create<fir::NegcOp>(getLoc(), genunbox(op.left()));
  }

  template <typename OpTy>
  mlir::Value createBinaryOp(const ExtValue &left, const ExtValue &right) {
    assert(fir::isUnboxedValue(left) && fir::isUnboxedValue(right));
    mlir::Value lhs = fir::getBase(left);
    mlir::Value rhs = fir::getBase(right);
    assert(lhs.getType() == rhs.getType() && "types must be the same");
    return builder.create<OpTy>(getLoc(), lhs, rhs);
  }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    ExtValue left = genval(ex.left());
    return createBinaryOp<OpTy>(left, genval(ex.right()));
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  ExtValue genval(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type< \
                      Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) { \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::arith::AddIOp)
  GENBIN(Add, Real, mlir::arith::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::arith::SubIOp)
  GENBIN(Subtract, Real, mlir::arith::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::arith::MulIOp)
  GENBIN(Multiply, Real, mlir::arith::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::arith::DivSIOp)
  GENBIN(Divide, Real, mlir::arith::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    mlir::Type ty = converter.genType(TC, KIND);
    mlir::Value lhs = genunbox(op.left());
    mlir::Value rhs = genunbox(op.right());
    return fir::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    mlir::Type ty = converter.genType(TC, KIND);
    mlir::Value lhs = genunbox(op.left());
    mlir::Value rhs = genunbox(op.right());
    return fir::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    mlir::Value realPartValue = genunbox(op.left());
    return fir::factory::Complex{builder, getLoc()}.createComplex(
        KIND, realPartValue, genunbox(op.right()));
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    ExtValue lhs = genval(op.left());
    ExtValue rhs = genval(op.right());
    const fir::CharBoxValue *lhsChar = lhs.getCharBox();
    const fir::CharBoxValue *rhsChar = rhs.getCharBox();
    if (lhsChar && rhsChar)
      return fir::factory::CharacterExprHelper{builder, getLoc()}
          .createConcatenate(*lhsChar, *rhsChar);
    TODO(getLoc(), "character array concatenate");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    mlir::Value lhs = genunbox(op.left());
    mlir::Value rhs = genunbox(op.right());
    switch (op.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return fir::genMax(builder, getLoc(),
                         llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Less:
      return fir::genMin(builder, getLoc(),
                         llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Equal:
      llvm_unreachable("Equal is not a valid ordering in this context");
    }
    llvm_unreachable("unknown ordering");
  }

  // Change the dynamic length information without actually changing the
  // underlying character storage.
  fir::ExtendedValue
  replaceScalarCharacterLength(const fir::ExtendedValue &scalarChar,
                               mlir::Value newLenValue) {
    mlir::Location loc = getLoc();
    const fir::CharBoxValue *charBox = scalarChar.getCharBox();
    if (!charBox)
      fir::emitFatalError(loc, "expected scalar character");
    mlir::Value charAddr = charBox->getAddr();
    auto charType =
        fir::unwrapPassByRefType(charAddr.getType()).cast<fir::CharacterType>();
    if (charType.hasConstantLen()) {
      // Erase previous constant length from the base type.
      fir::CharacterType::LenType newLen = fir::CharacterType::unknownLen();
      mlir::Type newCharTy = fir::CharacterType::get(
          builder.getContext(), charType.getFKind(), newLen);
      mlir::Type newType = fir::ReferenceType::get(newCharTy);
      charAddr = builder.createConvert(loc, newType, charAddr);
      return fir::CharBoxValue{charAddr, newLenValue};
    }
    return fir::CharBoxValue{charAddr, newLenValue};
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::SetLength<KIND> &x) {
    mlir::Value newLenValue = genunbox(x.right());
    fir::ExtendedValue lhs = gen(x.left());
    fir::factory::CharacterExprHelper charHelper(builder, getLoc());
    fir::CharBoxValue temp = charHelper.createCharacterTemp(
        charHelper.getCharacterType(fir::getBase(lhs).getType()), newLenValue);
    charHelper.createAssign(temp, lhs);
    return fir::ExtendedValue{temp};
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    return createCompareOp<mlir::arith::CmpIOp>(op,
                                                translateRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return createFltCmpOp<mlir::arith::CmpFOp>(
        op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return createFltCmpOp<fir::CmpcOp>(op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &op) {
    return createCharCompare(op, translateRelational(op.opr));
  }

  ExtValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  ExtValue
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    mlir::Type ty = converter.genType(TC1, KIND);
    auto fromExpr = genval(convert.left());
    auto loc = getLoc();
    return fromExpr.match(
        [&](const fir::CharBoxValue &boxchar) -> ExtValue {
          if constexpr (TC1 == Fortran::common::TypeCategory::Character &&
                        TC2 == TC1) {
            // Use char_convert. Each code point is translated from a
            // narrower/wider encoding to the target encoding. For example, 'A'
            // may be translated from 0x41 : i8 to 0x0041 : i16. The symbol
            // for euro (0x20AC : i16) may be translated from a wide character
            // to "0xE2 0x82 0xAC" : UTF-8.
            mlir::Value bufferSize = boxchar.getLen();
            auto kindMap = builder.getKindMap();
            mlir::Value boxCharAddr = boxchar.getAddr();
            auto fromTy = boxCharAddr.getType();
            if (auto charTy = fromTy.dyn_cast<fir::CharacterType>()) {
              // boxchar is a value, not a variable. Turn it into a temporary.
              // As a value, it ought to have a constant LEN value.
              assert(charTy.hasConstantLen() && "must have constant length");
              mlir::Value tmp = builder.createTemporary(loc, charTy);
              builder.create<fir::StoreOp>(loc, boxCharAddr, tmp);
              boxCharAddr = tmp;
            }
            auto fromBits =
                kindMap.getCharacterBitsize(fir::unwrapRefType(fromTy)
                                                .cast<fir::CharacterType>()
                                                .getFKind());
            auto toBits = kindMap.getCharacterBitsize(
                ty.cast<fir::CharacterType>().getFKind());
            if (toBits < fromBits) {
              // Scale by relative ratio to give a buffer of the same length.
              auto ratio = builder.createIntegerConstant(
                  loc, bufferSize.getType(), fromBits / toBits);
              bufferSize =
                  builder.create<mlir::arith::MulIOp>(loc, bufferSize, ratio);
            }
            auto dest = builder.create<fir::AllocaOp>(
                loc, ty, mlir::ValueRange{bufferSize});
            builder.create<fir::CharConvertOp>(loc, boxCharAddr,
                                               boxchar.getLen(), dest);
            return fir::CharBoxValue{dest, boxchar.getLen()};
          } else {
            fir::emitFatalError(
                loc, "unsupported evaluate::Convert between CHARACTER type "
                     "category and non-CHARACTER category");
          }
        },
        [&](const fir::UnboxedValue &value) -> ExtValue {
          return builder.convertWithSemantics(loc, ty, value);
        },
        [&](auto &) -> ExtValue {
          fir::emitFatalError(loc, "unsupported evaluate::Convert");
        });
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    ExtValue input = genval(op.left());
    mlir::Value base = fir::getBase(input);
    mlir::Value newBase =
        builder.create<fir::NoReassocOp>(getLoc(), base.getType(), base);
    return fir::substBase(input, newBase);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Not<KIND> &op) {
    mlir::Value logical = genunbox(op.left());
    mlir::Value one = genBoolConstant(true);
    mlir::Value val =
        builder.createConvert(getLoc(), builder.getI1Type(), logical);
    return builder.create<mlir::arith::XOrIOp>(getLoc(), val, one);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    mlir::IntegerType i1Type = builder.getI1Type();
    mlir::Value slhs = genunbox(op.left());
    mlir::Value srhs = genunbox(op.right());
    mlir::Value lhs = builder.createConvert(getLoc(), i1Type, slhs);
    mlir::Value rhs = builder.createConvert(getLoc(), i1Type, srhs);
    switch (op.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryOp<mlir::arith::AndIOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryOp<mlir::arith::OrIOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::eq, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::ne, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Not:
      // lib/evaluate expression for .NOT. is Fortran::evaluate::Not<KIND>.
      llvm_unreachable(".NOT. is not a binary operator");
    }
    llvm_unreachable("unhandled logical operation");
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    return Fortran::lower::convertConstant(
        converter, getLoc(), con,
        /*outlineBigConstantsInReadOnlyMemory=*/!inInitializer);
  }

  fir::ExtendedValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    if (auto ctor = con.GetScalarValue())
      return genval(*ctor);
    return Fortran::lower::convertConstant(
        converter, getLoc(), con,
        /*outlineBigConstantsInReadOnlyMemory=*/false);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    fir::emitFatalError(getLoc(), "array constructor: should not reach here");
  }

  ExtValue gen(const Fortran::evaluate::ComplexPart &x) {
    mlir::Location loc = getLoc();
    auto idxTy = builder.getI32Type();
    ExtValue exv = gen(x.complex());
    mlir::Value base = fir::getBase(exv);
    fir::factory::Complex helper{builder, loc};
    mlir::Type eleTy =
        helper.getComplexPartType(fir::dyn_cast_ptrEleTy(base.getType()));
    mlir::Value offset = builder.createIntegerConstant(
        loc, idxTy,
        x.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    mlir::Value result = builder.create<fir::CoordinateOp>(
        loc, builder.getRefType(eleTy), base, mlir::ValueRange{offset});
    return {result};
  }
  ExtValue genval(const Fortran::evaluate::ComplexPart &x) {
    return genLoad(gen(x));
  }

  /// Reference to a substring.
  ExtValue gen(const Fortran::evaluate::Substring &s) {
    // Get base string
    auto baseString = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &x) { return gen(x); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &p)
                -> ExtValue {
              if (std::optional<std::string> str = p->AsString())
                return fir::factory::createStringLiteral(builder, getLoc(),
                                                         *str);
              // TODO: convert StaticDataObject to Constant<T> and use normal
              // constant path. Beware that StaticDataObject data() takes into
              // account build machine endianness.
              TODO(getLoc(),
                   "StaticDataObject::Pointer substring with kind > 1");
            },
        },
        s.parent());
    llvm::SmallVector<mlir::Value> bounds;
    mlir::Value lower = genunbox(s.lower());
    bounds.push_back(lower);
    if (Fortran::evaluate::MaybeExtentExpr upperBound = s.upper()) {
      mlir::Value upper = genunbox(*upperBound);
      bounds.push_back(upper);
    }
    fir::factory::CharacterExprHelper charHelper{builder, getLoc()};
    return baseString.match(
        [&](const fir::CharBoxValue &x) -> ExtValue {
          return charHelper.createSubstring(x, bounds);
        },
        [&](const fir::CharArrayBoxValue &) -> ExtValue {
          fir::emitFatalError(
              getLoc(),
              "array substring should be handled in array expression");
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(getLoc(), "substring base is not a CharBox");
        });
  }

  /// The value of a substring.
  ExtValue genval(const Fortran::evaluate::Substring &ss) {
    // FIXME: why is the value of a substring being lowered the same as the
    // address of a substring?
    return gen(ss);
  }

  ExtValue genval(const Fortran::evaluate::Subscript &subs) {
    if (auto *s = std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
            &subs.u)) {
      if (s->value().Rank() > 0)
        fir::emitFatalError(getLoc(), "vector subscript is not scalar");
      return {genval(s->value())};
    }
    fir::emitFatalError(getLoc(), "subscript triple notation is not scalar");
  }
  ExtValue genSubscript(const Fortran::evaluate::Subscript &subs) {
    return genval(subs);
  }

  ExtValue gen(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  ExtValue genval(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return genval(x); }, dref.u);
  }

  // Helper function to turn the Component structure into a list of nested
  // components, ordered from largest/leftmost to smallest/rightmost:
  //  - where only the smallest/rightmost item may be allocatable or a pointer
  //    (nested allocatable/pointer components require nested coordinate_of ops)
  //  - that does not contain any parent components
  //    (the front end places parent components directly in the object)
  // Return the object used as the base coordinate for the component chain.
  static Fortran::evaluate::DataRef const *
  reverseComponents(const Fortran::evaluate::Component &cmpt,
                    std::list<const Fortran::evaluate::Component *> &list) {
    if (!getLastSym(cmpt).test(Fortran::semantics::Symbol::Flag::ParentComp))
      list.push_front(&cmpt);
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Component &x) {
              if (Fortran::semantics::IsAllocatableOrPointer(getLastSym(x)))
                return &cmpt.base();
              return reverseComponents(x, list);
            },
            [&](auto &) { return &cmpt.base(); },
        },
        cmpt.base().u);
  }

  // Return the coordinate of the component reference
  ExtValue genComponent(const Fortran::evaluate::Component &cmpt) {
    std::list<const Fortran::evaluate::Component *> list;
    const Fortran::evaluate::DataRef *base = reverseComponents(cmpt, list);
    llvm::SmallVector<mlir::Value> coorArgs;
    ExtValue obj = gen(*base);
    mlir::Type ty = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(obj).getType());
    mlir::Location loc = getLoc();
    auto fldTy = fir::FieldType::get(&converter.getMLIRContext());
    // FIXME: need to thread the LEN type parameters here.
    for (const Fortran::evaluate::Component *field : list) {
      auto recTy = ty.cast<fir::RecordType>();
      const Fortran::semantics::Symbol &sym = getLastSym(*field);
      llvm::StringRef name = toStringRef(sym.name());
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(
          loc, fldTy, name, recTy, fir::getTypeParams(obj)));
      ty = recTy.getType(name);
    }
    // If parent component is referred then it has no coordinate argument.
    if (coorArgs.size() == 0)
      return obj;
    ty = builder.getRefType(ty);
    return fir::factory::componentToExtendedValue(
        builder, loc,
        builder.create<fir::CoordinateOp>(loc, ty, fir::getBase(obj),
                                          coorArgs));
  }

  ExtValue gen(const Fortran::evaluate::Component &cmpt) {
    // Components may be pointer or allocatable. In the gen() path, the mutable
    // aspect is lost to simplify handling on the client side. To retain the
    // mutable aspect, genMutableBoxValue should be used.
    return genComponent(cmpt).match(
        [&](const fir::MutableBoxValue &mutableBox) {
          return fir::factory::genMutableBoxRead(builder, getLoc(), mutableBox);
        },
        [](auto &box) -> ExtValue { return box; });
  }

  ExtValue genval(const Fortran::evaluate::Component &cmpt) {
    return genLoad(gen(cmpt));
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  mlir::Type genSubType(mlir::Type arrTy, unsigned dims) {
    mlir::Type unwrapTy = fir::dyn_cast_ptrOrBoxEleTy(arrTy);
    assert(unwrapTy && "must be a pointer or box type");
    auto seqTy = unwrapTy.cast<fir::SequenceType>();
    llvm::ArrayRef<int64_t> shape = seqTy.getShape();
    assert(shape.size() > 0 && "removing columns for sequence sans shape");
    assert(dims <= shape.size() && "removing more columns than exist");
    fir::SequenceType::Shape newBnds;
    // follow Fortran semantics and remove columns (from right)
    std::size_t e = shape.size() - dims;
    for (decltype(e) i = 0; i < e; ++i)
      newBnds.push_back(shape[i]);
    if (!newBnds.empty())
      return fir::SequenceType::get(newBnds, seqTy.getEleTy());
    return seqTy.getEleTy();
  }

  // Generate the code for a Bound value.
  ExtValue genval(const Fortran::semantics::Bound &bound) {
    if (bound.isExplicit()) {
      Fortran::semantics::MaybeSubscriptIntExpr sub = bound.GetExplicit();
      if (sub.has_value())
        return genval(*sub);
      return genIntegerConstant<8>(builder.getContext(), 1);
    }
    TODO(getLoc(), "non explicit semantics::Bound implementation");
  }

  static bool isSlice(const Fortran::evaluate::ArrayRef &aref) {
    for (const Fortran::evaluate::Subscript &sub : aref.subscript())
      if (std::holds_alternative<Fortran::evaluate::Triplet>(sub.u))
        return true;
    return false;
  }

  /// Lower an ArrayRef to a fir.coordinate_of given its lowered base.
  ExtValue genCoordinateOp(const ExtValue &array,
                           const Fortran::evaluate::ArrayRef &aref) {
    mlir::Location loc = getLoc();
    // References to array of rank > 1 with non constant shape that are not
    // fir.box must be collapsed into an offset computation in lowering already.
    // The same is needed with dynamic length character arrays of all ranks.
    mlir::Type baseType =
        fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(array).getType());
    if ((array.rank() > 1 && fir::hasDynamicSize(baseType)) ||
        fir::characterWithDynamicLen(fir::unwrapSequenceType(baseType)))
      if (!array.getBoxOf<fir::BoxValue>())
        return genOffsetAndCoordinateOp(array, aref);
    // Generate a fir.coordinate_of with zero based array indexes.
    llvm::SmallVector<mlir::Value> args;
    for (const auto &subsc : llvm::enumerate(aref.subscript())) {
      ExtValue subVal = genSubscript(subsc.value());
      assert(fir::isUnboxedValue(subVal) && "subscript must be simple scalar");
      mlir::Value val = fir::getBase(subVal);
      mlir::Type ty = val.getType();
      mlir::Value lb = getLBound(array, subsc.index(), ty);
      args.push_back(builder.create<mlir::arith::SubIOp>(loc, ty, val, lb));
    }
    mlir::Value base = fir::getBase(array);
    mlir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(base.getType());
    if (auto classTy = eleTy.dyn_cast<fir::ClassType>())
      eleTy = classTy.getEleTy();
    auto seqTy = eleTy.cast<fir::SequenceType>();
    assert(args.size() == seqTy.getDimension());
    mlir::Type ty = builder.getRefType(seqTy.getEleTy());
    auto addr = builder.create<fir::CoordinateOp>(loc, ty, base, args);
    return fir::factory::arrayElementToExtendedValue(builder, loc, array, addr);
  }

  /// Lower an ArrayRef to a fir.coordinate_of using an element offset instead
  /// of array indexes.
  /// This generates offset computation from the indexes and length parameters,
  /// and use the offset to access the element with a fir.coordinate_of. This
  /// must only be used if it is not possible to generate a normal
  /// fir.coordinate_of using array indexes (i.e. when the shape information is
  /// unavailable in the IR).
  ExtValue genOffsetAndCoordinateOp(const ExtValue &array,
                                    const Fortran::evaluate::ArrayRef &aref) {
    mlir::Location loc = getLoc();
    mlir::Value addr = fir::getBase(array);
    mlir::Type arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    mlir::Type seqTy = builder.getRefType(builder.getVarLenSeqTy(eleTy));
    mlir::Type refTy = builder.getRefType(eleTy);
    mlir::Value base = builder.createConvert(loc, seqTy, addr);
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto getLB = [&](const auto &arr, unsigned dim) -> mlir::Value {
      return arr.getLBounds().empty() ? one : arr.getLBounds()[dim];
    };
    auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
      mlir::Value total = zero;
      assert(arr.getExtents().size() == aref.subscript().size());
      delta = builder.createConvert(loc, idxTy, delta);
      unsigned dim = 0;
      for (auto [ext, sub] : llvm::zip(arr.getExtents(), aref.subscript())) {
        ExtValue subVal = genSubscript(sub);
        assert(fir::isUnboxedValue(subVal));
        mlir::Value val =
            builder.createConvert(loc, idxTy, fir::getBase(subVal));
        mlir::Value lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
        mlir::Value diff = builder.create<mlir::arith::SubIOp>(loc, val, lb);
        mlir::Value prod =
            builder.create<mlir::arith::MulIOp>(loc, delta, diff);
        total = builder.create<mlir::arith::AddIOp>(loc, prod, total);
        if (ext)
          delta = builder.create<mlir::arith::MulIOp>(loc, delta, ext);
        ++dim;
      }
      mlir::Type origRefTy = refTy;
      if (fir::factory::CharacterExprHelper::isCharacterScalar(refTy)) {
        fir::CharacterType chTy =
            fir::factory::CharacterExprHelper::getCharacterType(refTy);
        if (fir::characterWithDynamicLen(chTy)) {
          mlir::MLIRContext *ctx = builder.getContext();
          fir::KindTy kind =
              fir::factory::CharacterExprHelper::getCharacterKind(chTy);
          fir::CharacterType singleTy =
              fir::CharacterType::getSingleton(ctx, kind);
          refTy = builder.getRefType(singleTy);
          mlir::Type seqRefTy =
              builder.getRefType(builder.getVarLenSeqTy(singleTy));
          base = builder.createConvert(loc, seqRefTy, base);
        }
      }
      auto coor = builder.create<fir::CoordinateOp>(
          loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
      // Convert to expected, original type after address arithmetic.
      return builder.createConvert(loc, origRefTy, coor);
    };
    return array.match(
        [&](const fir::ArrayBoxValue &arr) -> ExtValue {
          // FIXME: this check can be removed when slicing is implemented
          if (isSlice(aref))
            fir::emitFatalError(
                getLoc(),
                "slice should be handled in array expression context");
          return genFullDim(arr, one);
        },
        [&](const fir::CharArrayBoxValue &arr) -> ExtValue {
          mlir::Value delta = arr.getLen();
          // If the length is known in the type, fir.coordinate_of will
          // already take the length into account.
          if (fir::factory::CharacterExprHelper::hasConstantLengthInType(arr))
            delta = one;
          return fir::CharBoxValue(genFullDim(arr, delta), arr.getLen());
        },
        [&](const fir::BoxValue &arr) -> ExtValue {
          // CoordinateOp for BoxValue is not generated here. The dimensions
          // must be kept in the fir.coordinate_op so that potential fir.box
          // strides can be applied by codegen.
          fir::emitFatalError(
              loc, "internal: BoxValue in dim-collapsed fir.coordinate_of");
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "internal: array processing failed");
        });
  }

  /// Lower an ArrayRef to a fir.array_coor.
  ExtValue genArrayCoorOp(const ExtValue &exv,
                          const Fortran::evaluate::ArrayRef &aref) {
    mlir::Location loc = getLoc();
    mlir::Value addr = fir::getBase(exv);
    mlir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
    mlir::Type eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    mlir::Type refTy = builder.getRefType(eleTy);
    mlir::IndexType idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> arrayCoorArgs;
    // The ArrayRef is expected to be scalar here, arrays are handled in array
    // expression lowering. So no vector subscript or triplet is expected here.
    for (const auto &sub : aref.subscript()) {
      ExtValue subVal = genSubscript(sub);
      assert(fir::isUnboxedValue(subVal));
      arrayCoorArgs.push_back(
          builder.createConvert(loc, idxTy, fir::getBase(subVal)));
    }
    mlir::Value shape = builder.createShape(loc, exv);
    mlir::Value elementAddr = builder.create<fir::ArrayCoorOp>(
        loc, refTy, addr, shape, /*slice=*/mlir::Value{}, arrayCoorArgs,
        fir::getTypeParams(exv));
    return fir::factory::arrayElementToExtendedValue(builder, loc, exv,
                                                     elementAddr);
  }

  /// Return the coordinate of the array reference.
  ExtValue gen(const Fortran::evaluate::ArrayRef &aref) {
    ExtValue base = aref.base().IsSymbol() ? gen(getFirstSym(aref.base()))
                                           : gen(aref.base().GetComponent());
    // Check for command-line override to use array_coor op.
    if (generateArrayCoordinate)
      return genArrayCoorOp(base, aref);
    // Otherwise, use coordinate_of op.
    return genCoordinateOp(base, aref);
  }

  /// Return lower bounds of \p box in dimension \p dim. The returned value
  /// has type \ty.
  mlir::Value getLBound(const ExtValue &box, unsigned dim, mlir::Type ty) {
    assert(box.rank() > 0 && "must be an array");
    mlir::Location loc = getLoc();
    mlir::Value one = builder.createIntegerConstant(loc, ty, 1);
    mlir::Value lb = fir::factory::readLowerBound(builder, loc, box, dim, one);
    return builder.createConvert(loc, ty, lb);
  }

  ExtValue genval(const Fortran::evaluate::ArrayRef &aref) {
    return genLoad(gen(aref));
  }

  ExtValue gen(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genAddr(coref);
  }

  ExtValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genValue(coref);
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  ExtValue genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  mlir::Type genType(const Fortran::evaluate::DynamicType &dt) {
    if (dt.category() != Fortran::common::TypeCategory::Derived)
      return converter.genType(dt.category(), dt.kind());
    if (dt.IsUnlimitedPolymorphic())
      return mlir::NoneType::get(&converter.getMLIRContext());
    return converter.genType(dt.GetDerivedTypeSpec());
  }

  /// Lower a function reference
  template <typename A>
  ExtValue genFunctionRef(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    if (!funcRef.GetType().has_value())
      fir::emitFatalError(getLoc(), "a function must have a type");
    mlir::Type resTy = genType(*funcRef.GetType());
    return genProcedureRef(funcRef, {resTy});
  }

  /// Lower function call `funcRef` and return a reference to the resultant
  /// value. This is required for lowering expressions such as `f1(f2(v))`.
  template <typename A>
  ExtValue gen(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    ExtValue retVal = genFunctionRef(funcRef);
    mlir::Type resultType = converter.genType(toEvExpr(funcRef));
    return placeScalarValueInMemory(builder, getLoc(), retVal, resultType);
  }

  /// Helper to lower intrinsic arguments for inquiry intrinsic.
  ExtValue
  lowerIntrinsicArgumentAsInquired(const Fortran::lower::SomeExpr &expr) {
    if (Fortran::evaluate::IsAllocatableOrPointerObject(
            expr, converter.getFoldingContext()))
      return genMutableBoxValue(expr);
    /// Do not create temps for array sections whose properties only need to be
    /// inquired: create a descriptor that will be inquired.
    if (Fortran::evaluate::IsVariable(expr) && isArray(expr) &&
        !Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr))
      return lowerIntrinsicArgumentAsBox(expr);
    return gen(expr);
  }

  /// Helper to lower intrinsic arguments to a fir::BoxValue.
  /// It preserves all the non default lower bounds/non deferred length
  /// parameter information.
  ExtValue lowerIntrinsicArgumentAsBox(const Fortran::lower::SomeExpr &expr) {
    mlir::Location loc = getLoc();
    ExtValue exv = genBoxArg(expr);
    auto exvTy = fir::getBase(exv).getType();
    if (exvTy.isa<mlir::FunctionType>()) {
      auto boxProcTy = builder.getBoxProcType(exvTy.cast<mlir::FunctionType>());
      return builder.create<fir::EmboxProcOp>(loc, boxProcTy,
                                              fir::getBase(exv));
    }
    mlir::Value box = builder.createBox(loc, exv, exv.isPolymorphic());
    if (Fortran::lower::isParentComponent(expr)) {
      fir::ExtendedValue newExv =
          Fortran::lower::updateBoxForParentComponent(converter, box, expr);
      box = fir::getBase(newExv);
    }
    return fir::BoxValue(
        box, fir::factory::getNonDefaultLowerBounds(builder, loc, exv),
        fir::factory::getNonDeferredLenParams(exv));
  }

  /// Generate a call to a Fortran intrinsic or intrinsic module procedure.
  ExtValue genIntrinsicRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      std::optional<mlir::Type> resultType,
      std::optional<const Fortran::evaluate::SpecificIntrinsic> intrinsic =
          std::nullopt) {
    llvm::SmallVector<ExtValue> operands;

    std::string name =
        intrinsic ? intrinsic->name
                  : procRef.proc().GetSymbol()->GetUltimate().name().ToString();
    mlir::Location loc = getLoc();
    if (intrinsic && Fortran::lower::intrinsicRequiresCustomOptionalHandling(
                         procRef, *intrinsic, converter)) {
      using ExvAndPresence = std::pair<ExtValue, std::optional<mlir::Value>>;
      llvm::SmallVector<ExvAndPresence, 4> operands;
      auto prepareOptionalArg = [&](const Fortran::lower::SomeExpr &expr) {
        ExtValue optionalArg = lowerIntrinsicArgumentAsInquired(expr);
        mlir::Value isPresent =
            genActualIsPresentTest(builder, loc, optionalArg);
        operands.emplace_back(optionalArg, isPresent);
      };
      auto prepareOtherArg = [&](const Fortran::lower::SomeExpr &expr,
                                 fir::LowerIntrinsicArgAs lowerAs) {
        switch (lowerAs) {
        case fir::LowerIntrinsicArgAs::Value:
          operands.emplace_back(genval(expr), std::nullopt);
          return;
        case fir::LowerIntrinsicArgAs::Addr:
          operands.emplace_back(gen(expr), std::nullopt);
          return;
        case fir::LowerIntrinsicArgAs::Box:
          operands.emplace_back(lowerIntrinsicArgumentAsBox(expr),
                                std::nullopt);
          return;
        case fir::LowerIntrinsicArgAs::Inquired:
          operands.emplace_back(lowerIntrinsicArgumentAsInquired(expr),
                                std::nullopt);
          return;
        }
      };
      Fortran::lower::prepareCustomIntrinsicArgument(
          procRef, *intrinsic, resultType, prepareOptionalArg, prepareOtherArg,
          converter);

      auto getArgument = [&](std::size_t i, bool loadArg) -> ExtValue {
        if (loadArg && fir::conformsWithPassByRef(
                           fir::getBase(operands[i].first).getType()))
          return genLoad(operands[i].first);
        return operands[i].first;
      };
      auto isPresent = [&](std::size_t i) -> std::optional<mlir::Value> {
        return operands[i].second;
      };
      return Fortran::lower::lowerCustomIntrinsic(
          builder, loc, name, resultType, isPresent, getArgument,
          operands.size(), stmtCtx);
    }

    const fir::IntrinsicArgumentLoweringRules *argLowering =
        fir::getIntrinsicArgumentLowering(name);
    for (const auto &arg : llvm::enumerate(procRef.arguments())) {
      auto *expr =
          Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
      if (!expr) {
        // Absent optional.
        operands.emplace_back(fir::getAbsentIntrinsicArgument());
        continue;
      }
      if (!argLowering) {
        // No argument lowering instruction, lower by value.
        operands.emplace_back(genval(*expr));
        continue;
      }
      // Ad-hoc argument lowering handling.
      fir::ArgLoweringRule argRules =
          fir::lowerIntrinsicArgumentAs(*argLowering, arg.index());
      if (argRules.handleDynamicOptional &&
          Fortran::evaluate::MayBePassedAsAbsentOptional(
              *expr, converter.getFoldingContext())) {
        ExtValue optional = lowerIntrinsicArgumentAsInquired(*expr);
        mlir::Value isPresent = genActualIsPresentTest(builder, loc, optional);
        switch (argRules.lowerAs) {
        case fir::LowerIntrinsicArgAs::Value:
          operands.emplace_back(
              genOptionalValue(builder, loc, optional, isPresent));
          continue;
        case fir::LowerIntrinsicArgAs::Addr:
          operands.emplace_back(
              genOptionalAddr(builder, loc, optional, isPresent));
          continue;
        case fir::LowerIntrinsicArgAs::Box:
          operands.emplace_back(
              genOptionalBox(builder, loc, optional, isPresent));
          continue;
        case fir::LowerIntrinsicArgAs::Inquired:
          operands.emplace_back(optional);
          continue;
        }
        llvm_unreachable("bad switch");
      }
      switch (argRules.lowerAs) {
      case fir::LowerIntrinsicArgAs::Value:
        operands.emplace_back(genval(*expr));
        continue;
      case fir::LowerIntrinsicArgAs::Addr:
        operands.emplace_back(gen(*expr));
        continue;
      case fir::LowerIntrinsicArgAs::Box:
        operands.emplace_back(lowerIntrinsicArgumentAsBox(*expr));
        continue;
      case fir::LowerIntrinsicArgAs::Inquired:
        operands.emplace_back(lowerIntrinsicArgumentAsInquired(*expr));
        continue;
      }
      llvm_unreachable("bad switch");
    }
    // Let the intrinsic library lower the intrinsic procedure call
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name, resultType,
                                            operands, stmtCtx);
  }

  /// helper to detect statement functions
  static bool
  isStatementFunctionCall(const Fortran::evaluate::ProcedureRef &procRef) {
    if (const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol())
      if (const auto *details =
              symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
        return details->stmtFunction().has_value();
    return false;
  }

  /// Generate Statement function calls
  ExtValue genStmtFunctionRef(const Fortran::evaluate::ProcedureRef &procRef) {
    const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol();
    assert(symbol && "expected symbol in ProcedureRef of statement functions");
    const auto &details = symbol->get<Fortran::semantics::SubprogramDetails>();

    // Statement functions have their own scope, we just need to associate
    // the dummy symbols to argument expressions. They are no
    // optional/alternate return arguments. Statement functions cannot be
    // recursive (directly or indirectly) so it is safe to add dummy symbols to
    // the local map here.
    symMap.pushScope();
    for (auto [arg, bind] :
         llvm::zip(details.dummyArgs(), procRef.arguments())) {
      assert(arg && "alternate return in statement function");
      assert(bind && "optional argument in statement function");
      const auto *expr = bind->UnwrapExpr();
      // TODO: assumed type in statement function, that surprisingly seems
      // allowed, probably because nobody thought of restricting this usage.
      // gfortran/ifort compiles this.
      assert(expr && "assumed type used as statement function argument");
      // As per Fortran 2018 C1580, statement function arguments can only be
      // scalars, so just pass the box with the address. The only care is to
      // to use the dummy character explicit length if any instead of the
      // actual argument length (that can be bigger).
      if (const Fortran::semantics::DeclTypeSpec *type = arg->GetType())
        if (type->category() == Fortran::semantics::DeclTypeSpec::Character)
          if (const Fortran::semantics::MaybeIntExpr &lenExpr =
                  type->characterTypeSpec().length().GetExplicit()) {
            mlir::Value len = fir::getBase(genval(*lenExpr));
            // F2018 7.4.4.2 point 5.
            len = fir::factory::genMaxWithZero(builder, getLoc(), len);
            symMap.addSymbol(*arg,
                             replaceScalarCharacterLength(gen(*expr), len));
            continue;
          }
      symMap.addSymbol(*arg, gen(*expr));
    }

    // Explicitly map statement function host associated symbols to their
    // parent scope lowered symbol box.
    for (const Fortran::semantics::SymbolRef &sym :
         Fortran::evaluate::CollectSymbols(*details.stmtFunction()))
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>())
        if (!symMap.lookupSymbol(*sym))
          symMap.addSymbol(*sym, gen(details->symbol()));

    ExtValue result = genval(details.stmtFunction().value());
    LLVM_DEBUG(llvm::dbgs() << "stmt-function: " << result << '\n');
    symMap.popScope();
    return result;
  }

  /// Create a contiguous temporary array with the same shape,
  /// length parameters and type as mold. It is up to the caller to deallocate
  /// the temporary.
  ExtValue genArrayTempFromMold(const ExtValue &mold,
                                llvm::StringRef tempName) {
    mlir::Type type = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(mold).getType());
    assert(type && "expected descriptor or memory type");
    mlir::Location loc = getLoc();
    llvm::SmallVector<mlir::Value> extents =
        fir::factory::getExtents(loc, builder, mold);
    llvm::SmallVector<mlir::Value> allocMemTypeParams =
        fir::getTypeParams(mold);
    mlir::Value charLen;
    mlir::Type elementType = fir::unwrapSequenceType(type);
    if (auto charType = elementType.dyn_cast<fir::CharacterType>()) {
      charLen = allocMemTypeParams.empty()
                    ? fir::factory::readCharLen(builder, loc, mold)
                    : allocMemTypeParams[0];
      if (charType.hasDynamicLen() && allocMemTypeParams.empty())
        allocMemTypeParams.push_back(charLen);
    } else if (fir::hasDynamicSize(elementType)) {
      TODO(loc, "creating temporary for derived type with length parameters");
    }

    mlir::Value temp = builder.create<fir::AllocMemOp>(
        loc, type, tempName, allocMemTypeParams, extents);
    if (fir::unwrapSequenceType(type).isa<fir::CharacterType>())
      return fir::CharArrayBoxValue{temp, charLen, extents};
    return fir::ArrayBoxValue{temp, extents};
  }

  /// Copy \p source array into \p dest array. Both arrays must be
  /// conforming, but neither array must be contiguous.
  void genArrayCopy(ExtValue dest, ExtValue source) {
    return createSomeArrayAssignment(converter, dest, source, symMap, stmtCtx);
  }

  /// Lower a non-elemental procedure reference and read allocatable and pointer
  /// results into normal values.
  ExtValue genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                           std::optional<mlir::Type> resultType) {
    ExtValue res = genRawProcedureRef(procRef, resultType);
    // In most contexts, pointers and allocatable do not appear as allocatable
    // or pointer variable on the caller side (see 8.5.3 note 1 for
    // allocatables). The few context where this can happen must call
    // genRawProcedureRef directly.
    if (const auto *box = res.getBoxOf<fir::MutableBoxValue>())
      return fir::factory::genMutableBoxRead(builder, getLoc(), *box);
    return res;
  }

  /// Like genExtAddr, but ensure the address returned is a temporary even if \p
  /// expr is variable inside parentheses.
  ExtValue genTempExtAddr(const Fortran::lower::SomeExpr &expr) {
    // In general, genExtAddr might not create a temp for variable inside
    // parentheses to avoid creating array temporary in sub-expressions. It only
    // ensures the sub-expression is not re-associated with other parts of the
    // expression. In the call semantics, there is a difference between expr and
    // variable (see R1524). For expressions, a variable storage must not be
    // argument associated since it could be modified inside the call, or the
    // variable could also be modified by other means during the call.
    if (!isParenthesizedVariable(expr))
      return genExtAddr(expr);
    if (expr.Rank() > 0)
      return asArray(expr);
    mlir::Location loc = getLoc();
    return genExtValue(expr).match(
        [&](const fir::CharBoxValue &boxChar) -> ExtValue {
          return fir::factory::CharacterExprHelper{builder, loc}.createTempFrom(
              boxChar);
        },
        [&](const fir::UnboxedValue &v) -> ExtValue {
          mlir::Type type = v.getType();
          mlir::Value value = v;
          if (fir::isa_ref_type(type))
            value = builder.create<fir::LoadOp>(loc, value);
          mlir::Value temp = builder.createTemporary(loc, value.getType());
          builder.create<fir::StoreOp>(loc, value, temp);
          return temp;
        },
        [&](const fir::BoxValue &x) -> ExtValue {
          // Derived type scalar that may be polymorphic.
          if (fir::isPolymorphicType(fir::getBase(x).getType()))
            TODO(loc, "polymorphic array temporary");
          assert(!x.hasRank() && x.isDerived());
          if (x.isDerivedWithLenParameters())
            fir::emitFatalError(
                loc, "making temps for derived type with length parameters");
          // TODO: polymorphic aspects should be kept but for now the temp
          // created always has the declared type.
          mlir::Value var =
              fir::getBase(fir::factory::readBoxValue(builder, loc, x));
          auto value = builder.create<fir::LoadOp>(loc, var);
          mlir::Value temp = builder.createTemporary(loc, value.getType());
          builder.create<fir::StoreOp>(loc, value, temp);
          return temp;
        },
        [&](const fir::PolymorphicValue &p) -> ExtValue {
          mlir::Type type = p.getAddr().getType();
          mlir::Value value = p.getAddr();
          if (fir::isa_ref_type(type))
            value = builder.create<fir::LoadOp>(loc, value);
          mlir::Value temp = builder.createTemporary(loc, value.getType());
          builder.create<fir::StoreOp>(loc, value, temp);
          mlir::Value empty;
          mlir::ValueRange emptyRange;
          auto boxTy = fir::ClassType::get(value.getType());
          return builder.create<fir::EmboxOp>(loc, boxTy, temp, empty, empty,
                                              emptyRange, p.getSourceBox());
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "expr is not a scalar value");
        });
  }

  /// Helper structure to track potential copy-in of non contiguous variable
  /// argument into a contiguous temp. It is used to deallocate the temp that
  /// may have been created as well as to the copy-out from the temp to the
  /// variable after the call.
  struct CopyOutPair {
    ExtValue var;
    ExtValue temp;
    // Flag to indicate if the argument may have been modified by the
    // callee, in which case it must be copied-out to the variable.
    bool argMayBeModifiedByCall;
    // Optional boolean value that, if present and false, prevents
    // the copy-out and temp deallocation.
    std::optional<mlir::Value> restrictCopyAndFreeAtRuntime;
  };
  using CopyOutPairs = llvm::SmallVector<CopyOutPair, 4>;

  /// Helper to read any fir::BoxValue into other fir::ExtendedValue categories
  /// not based on fir.box.
  /// This will lose any non contiguous stride information and dynamic type and
  /// should only be called if \p exv is known to be contiguous or if its base
  /// address will be replaced by a contiguous one. If \p exv is not a
  /// fir::BoxValue, this is a no-op.
  ExtValue readIfBoxValue(const ExtValue &exv) {
    if (const auto *box = exv.getBoxOf<fir::BoxValue>())
      return fir::factory::readBoxValue(builder, getLoc(), *box);
    return exv;
  }

  /// Generate a contiguous temp to pass \p actualArg as argument \p arg. The
  /// creation of the temp and copy-in can be made conditional at runtime by
  /// providing a runtime boolean flag \p restrictCopyAtRuntime (in which case
  /// the temp and copy will only be made if the value is true at runtime).
  ExtValue genCopyIn(const ExtValue &actualArg,
                     const Fortran::lower::CallerInterface::PassedEntity &arg,
                     CopyOutPairs &copyOutPairs,
                     std::optional<mlir::Value> restrictCopyAtRuntime,
                     bool byValue) {
    const bool doCopyOut = !byValue && arg.mayBeModifiedByCall();
    llvm::StringRef tempName = byValue ? ".copy" : ".copyinout";
    mlir::Location loc = getLoc();
    bool isActualArgBox = fir::isa_box_type(fir::getBase(actualArg).getType());
    mlir::Value isContiguousResult;
    mlir::Type addrType = fir::HeapType::get(
        fir::unwrapPassByRefType(fir::getBase(actualArg).getType()));

    if (isActualArgBox) {
      // Check at runtime if the argument is contiguous so no copy is needed.
      isContiguousResult =
          fir::runtime::genIsContiguous(builder, loc, fir::getBase(actualArg));
    }

    auto doCopyIn = [&]() -> ExtValue {
      ExtValue temp = genArrayTempFromMold(actualArg, tempName);
      if (!arg.mayBeReadByCall()) {
        return temp;
      }
      if (!isActualArgBox || inlineCopyInOutForBoxes) {
        genArrayCopy(temp, actualArg);
        return temp;
      }

      // Generate Assign() call to copy data from the actualArg
      // to a temporary.
      mlir::Value destBox = fir::getBase(builder.createBox(loc, temp));
      mlir::Value boxRef = builder.createTemporary(loc, destBox.getType());
      builder.create<fir::StoreOp>(loc, destBox, boxRef);
      fir::runtime::genAssign(builder, loc, boxRef, fir::getBase(actualArg));
      return temp;
    };

    auto noCopy = [&]() {
      mlir::Value box = fir::getBase(actualArg);
      mlir::Value boxAddr = builder.create<fir::BoxAddrOp>(loc, addrType, box);
      builder.create<fir::ResultOp>(loc, boxAddr);
    };

    auto combinedCondition = [&]() {
      if (isActualArgBox) {
        mlir::Value zero =
            builder.createIntegerConstant(loc, builder.getI1Type(), 0);
        mlir::Value notContiguous = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, isContiguousResult, zero);
        if (!restrictCopyAtRuntime) {
          restrictCopyAtRuntime = notContiguous;
        } else {
          mlir::Value cond = builder.create<mlir::arith::AndIOp>(
              loc, *restrictCopyAtRuntime, notContiguous);
          restrictCopyAtRuntime = cond;
        }
      }
    };

    if (!restrictCopyAtRuntime) {
      if (isActualArgBox) {
        // isContiguousResult = genIsContiguousCall();
        mlir::Value addr =
            builder
                .genIfOp(loc, {addrType}, isContiguousResult,
                         /*withElseRegion=*/true)
                .genThen([&]() { noCopy(); })
                .genElse([&] {
                  ExtValue temp = doCopyIn();
                  builder.create<fir::ResultOp>(loc, fir::getBase(temp));
                })
                .getResults()[0];
        fir::ExtendedValue temp =
            fir::substBase(readIfBoxValue(actualArg), addr);
        combinedCondition();
        copyOutPairs.emplace_back(
            CopyOutPair{actualArg, temp, doCopyOut, restrictCopyAtRuntime});
        return temp;
      }

      ExtValue temp = doCopyIn();
      copyOutPairs.emplace_back(CopyOutPair{actualArg, temp, doCopyOut, {}});
      return temp;
    }

    // Otherwise, need to be careful to only copy-in if allowed at runtime.
    mlir::Value addr =
        builder
            .genIfOp(loc, {addrType}, *restrictCopyAtRuntime,
                     /*withElseRegion=*/true)
            .genThen([&]() {
              if (isActualArgBox) {
                // isContiguousResult = genIsContiguousCall();
                // Avoid copyin if the argument is contiguous at runtime.
                mlir::Value addr1 =
                    builder
                        .genIfOp(loc, {addrType}, isContiguousResult,
                                 /*withElseRegion=*/true)
                        .genThen([&]() { noCopy(); })
                        .genElse([&]() {
                          ExtValue temp = doCopyIn();
                          builder.create<fir::ResultOp>(loc,
                                                        fir::getBase(temp));
                        })
                        .getResults()[0];
                builder.create<fir::ResultOp>(loc, addr1);
              } else {
                ExtValue temp = doCopyIn();
                builder.create<fir::ResultOp>(loc, fir::getBase(temp));
              }
            })
            .genElse([&]() {
              mlir::Value nullPtr = builder.createNullConstant(loc, addrType);
              builder.create<fir::ResultOp>(loc, nullPtr);
            })
            .getResults()[0];
    // Associate the temp address with actualArg lengths and extents if a
    // temporary is generated. Otherwise the same address is associated.
    fir::ExtendedValue temp = fir::substBase(readIfBoxValue(actualArg), addr);
    combinedCondition();
    copyOutPairs.emplace_back(
        CopyOutPair{actualArg, temp, doCopyOut, restrictCopyAtRuntime});
    return temp;
  }

  /// Generate copy-out if needed and free the temporary for an argument that
  /// has been copied-in into a contiguous temp.
  void genCopyOut(const CopyOutPair &copyOutPair) {
    mlir::Location loc = getLoc();
    bool isActualArgBox =
        fir::isa_box_type(fir::getBase(copyOutPair.var).getType());
    auto doCopyOut = [&]() {
      if (!copyOutPair.argMayBeModifiedByCall) {
        return;
      }
      if (!isActualArgBox || inlineCopyInOutForBoxes) {
        genArrayCopy(copyOutPair.var, copyOutPair.temp);
        return;
      }
      // Generate Assign() call to copy data from the temporary
      // to the actualArg. Note that in case the actual argument
      // is ALLOCATABLE/POINTER the Assign() implementation
      // should not engage its reallocation, because the temporary
      // is rank, shape and type compatible with it.
      mlir::Value srcBox =
          fir::getBase(builder.createBox(loc, copyOutPair.temp));
      mlir::Value destBox =
          fir::getBase(builder.createBox(loc, copyOutPair.var));
      mlir::Value destBoxRef = builder.createTemporary(loc, destBox.getType());
      builder.create<fir::StoreOp>(loc, destBox, destBoxRef);
      fir::runtime::genAssign(builder, loc, destBoxRef, srcBox);
    };
    if (!copyOutPair.restrictCopyAndFreeAtRuntime) {
      doCopyOut();
      builder.create<fir::FreeMemOp>(loc, fir::getBase(copyOutPair.temp));
      return;
    }

    builder.genIfThen(loc, *copyOutPair.restrictCopyAndFreeAtRuntime)
        .genThen([&]() {
          doCopyOut();
          builder.create<fir::FreeMemOp>(loc, fir::getBase(copyOutPair.temp));
        })
        .end();
  }

  /// Lower a designator to a variable that may be absent at runtime into an
  /// ExtendedValue where all the properties (base address, shape and length
  /// parameters) can be safely read (set to zero if not present). It also
  /// returns a boolean mlir::Value telling if the variable is present at
  /// runtime.
  /// This is useful to later be able to do conditional copy-in/copy-out
  /// or to retrieve the base address without having to deal with the case
  /// where the actual may be an absent fir.box.
  std::pair<ExtValue, mlir::Value>
  prepareActualThatMayBeAbsent(const Fortran::lower::SomeExpr &expr) {
    mlir::Location loc = getLoc();
    if (Fortran::evaluate::IsAllocatableOrPointerObject(
            expr, converter.getFoldingContext())) {
      // Fortran 2018 15.5.2.12 point 1: If unallocated/disassociated,
      // it is as if the argument was absent. The main care here is to
      // not do a copy-in/copy-out because the temp address, even though
      // pointing to a null size storage, would not be a nullptr and
      // therefore the argument would not be considered absent on the
      // callee side. Note: if wholeSymbol is optional, it cannot be
      // absent as per 15.5.2.12 point 7. and 8. We rely on this to
      // un-conditionally read the allocatable/pointer descriptor here.
      fir::MutableBoxValue mutableBox = genMutableBoxValue(expr);
      mlir::Value isPresent = fir::factory::genIsAllocatedOrAssociatedTest(
          builder, loc, mutableBox);
      fir::ExtendedValue actualArg =
          fir::factory::genMutableBoxRead(builder, loc, mutableBox);
      return {actualArg, isPresent};
    }
    // Absent descriptor cannot be read. To avoid any issue in
    // copy-in/copy-out, and when retrieving the address/length
    // create an descriptor pointing to a null address here if the
    // fir.box is absent.
    ExtValue actualArg = gen(expr);
    mlir::Value actualArgBase = fir::getBase(actualArg);
    mlir::Value isPresent = builder.create<fir::IsPresentOp>(
        loc, builder.getI1Type(), actualArgBase);
    if (!actualArgBase.getType().isa<fir::BoxType>())
      return {actualArg, isPresent};
    ExtValue safeToReadBox =
        absentBoxToUnallocatedBox(builder, loc, actualArg, isPresent);
    return {safeToReadBox, isPresent};
  }

  /// Create a temp on the stack for scalar actual arguments that may be absent
  /// at runtime, but must be passed via a temp if they are presents.
  fir::ExtendedValue
  createScalarTempForArgThatMayBeAbsent(ExtValue actualArg,
                                        mlir::Value isPresent) {
    mlir::Location loc = getLoc();
    mlir::Type type = fir::unwrapRefType(fir::getBase(actualArg).getType());
    if (fir::isDerivedWithLenParameters(actualArg))
      TODO(loc, "parametrized derived type optional scalar argument copy-in");
    if (const fir::CharBoxValue *charBox = actualArg.getCharBox()) {
      mlir::Value len = charBox->getLen();
      mlir::Value zero = builder.createIntegerConstant(loc, len.getType(), 0);
      len = builder.create<mlir::arith::SelectOp>(loc, isPresent, len, zero);
      mlir::Value temp = builder.createTemporary(
          loc, type, /*name=*/{},
          /*shape=*/{}, mlir::ValueRange{len},
          llvm::ArrayRef<mlir::NamedAttribute>{
              Fortran::lower::getAdaptToByRefAttr(builder)});
      return fir::CharBoxValue{temp, len};
    }
    assert((fir::isa_trivial(type) || type.isa<fir::RecordType>()) &&
           "must be simple scalar");
    return builder.createTemporary(
        loc, type,
        llvm::ArrayRef<mlir::NamedAttribute>{
            Fortran::lower::getAdaptToByRefAttr(builder)});
  }

  template <typename A>
  bool isCharacterType(const A &exp) {
    if (auto type = exp.GetType())
      return type->category() == Fortran::common::TypeCategory::Character;
    return false;
  }

  /// Lower an actual argument that must be passed via an address.
  /// This generates of the copy-in/copy-out if the actual is not contiguous, or
  /// the creation of the temp if the actual is a variable and \p byValue is
  /// true. It handles the cases where the actual may be absent, and all of the
  /// copying has to be conditional at runtime.
  /// If the actual argument may be dynamically absent, return an additional
  /// boolean mlir::Value that if true means that the actual argument is
  /// present.
  std::pair<ExtValue, std::optional<mlir::Value>>
  prepareActualToBaseAddressLike(
      const Fortran::lower::SomeExpr &expr,
      const Fortran::lower::CallerInterface::PassedEntity &arg,
      CopyOutPairs &copyOutPairs, bool byValue) {
    mlir::Location loc = getLoc();
    const bool isArray = expr.Rank() > 0;
    const bool actualArgIsVariable = Fortran::evaluate::IsVariable(expr);
    // It must be possible to modify VALUE arguments on the callee side, even
    // if the actual argument is a literal or named constant. Hence, the
    // address of static storage must not be passed in that case, and a copy
    // must be made even if this is not a variable.
    // Note: isArray should be used here, but genBoxArg already creates copies
    // for it, so do not duplicate the copy until genBoxArg behavior is changed.
    const bool isStaticConstantByValue =
        byValue && Fortran::evaluate::IsActuallyConstant(expr) &&
        (isCharacterType(expr));
    const bool variableNeedsCopy =
        actualArgIsVariable &&
        (byValue || (isArray && !Fortran::evaluate::IsSimplyContiguous(
                                    expr, converter.getFoldingContext())));
    const bool needsCopy = isStaticConstantByValue || variableNeedsCopy;
    auto [argAddr, isPresent] =
        [&]() -> std::pair<ExtValue, std::optional<mlir::Value>> {
      if (!actualArgIsVariable && !needsCopy)
        // Actual argument is not a variable. Make sure a variable address is
        // not passed.
        return {genTempExtAddr(expr), std::nullopt};
      ExtValue baseAddr;
      if (arg.isOptional() && Fortran::evaluate::MayBePassedAsAbsentOptional(
                                  expr, converter.getFoldingContext())) {
        auto [actualArgBind, isPresent] = prepareActualThatMayBeAbsent(expr);
        const ExtValue &actualArg = actualArgBind;
        if (!needsCopy)
          return {actualArg, isPresent};

        if (isArray)
          return {genCopyIn(actualArg, arg, copyOutPairs, isPresent, byValue),
                  isPresent};
        // Scalars, create a temp, and use it conditionally at runtime if
        // the argument is present.
        ExtValue temp =
            createScalarTempForArgThatMayBeAbsent(actualArg, isPresent);
        mlir::Type tempAddrTy = fir::getBase(temp).getType();
        mlir::Value selectAddr =
            builder
                .genIfOp(loc, {tempAddrTy}, isPresent,
                         /*withElseRegion=*/true)
                .genThen([&]() {
                  fir::factory::genScalarAssignment(builder, loc, temp,
                                                    actualArg);
                  builder.create<fir::ResultOp>(loc, fir::getBase(temp));
                })
                .genElse([&]() {
                  mlir::Value absent =
                      builder.create<fir::AbsentOp>(loc, tempAddrTy);
                  builder.create<fir::ResultOp>(loc, absent);
                })
                .getResults()[0];
        return {fir::substBase(temp, selectAddr), isPresent};
      }
      // Actual cannot be absent, the actual argument can safely be
      // copied-in/copied-out without any care if needed.
      if (isArray) {
        ExtValue box = genBoxArg(expr);
        if (needsCopy)
          return {genCopyIn(box, arg, copyOutPairs,
                            /*restrictCopyAtRuntime=*/std::nullopt, byValue),
                  std::nullopt};
        // Contiguous: just use the box we created above!
        // This gets "unboxed" below, if needed.
        return {box, std::nullopt};
      }
      // Actual argument is a non-optional, non-pointer, non-allocatable
      // scalar.
      ExtValue actualArg = genExtAddr(expr);
      if (needsCopy)
        return {createInMemoryScalarCopy(builder, loc, actualArg),
                std::nullopt};
      return {actualArg, std::nullopt};
    }();
    // Scalar and contiguous expressions may be lowered to a fir.box,
    // either to account for potential polymorphism, or because lowering
    // did not account for some contiguity hints.
    // Here, polymorphism does not matter (an entity of the declared type
    // is passed, not one of the dynamic type), and the expr is known to
    // be simply contiguous, so it is safe to unbox it and pass the
    // address without making a copy.
    return {readIfBoxValue(argAddr), isPresent};
  }

  /// Lower a non-elemental procedure reference.
  ExtValue genRawProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                              std::optional<mlir::Type> resultType) {
    mlir::Location loc = getLoc();
    if (isElementalProcWithArrayArgs(procRef))
      fir::emitFatalError(loc, "trying to lower elemental procedure with array "
                               "arguments as normal procedure");

    if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
            procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, resultType, *intrinsic);

    if (Fortran::lower::isIntrinsicModuleProcRef(procRef))
      return genIntrinsicRef(procRef, resultType);

    if (isStatementFunctionCall(procRef))
      return genStmtFunctionRef(procRef);

    Fortran::lower::CallerInterface caller(procRef, converter);
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    llvm::SmallVector<fir::MutableBoxValue> mutableModifiedByCall;
    // List of <var, temp> where temp must be copied into var after the call.
    CopyOutPairs copyOutPairs;

    mlir::FunctionType callSiteType = caller.genFunctionType();

    // Lower the actual arguments and map the lowered values to the dummy
    // arguments.
    for (const Fortran::lower::CallInterface<
             Fortran::lower::CallerInterface>::PassedEntity &arg :
         caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      mlir::Type argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        caller.placeInput(arg, builder.genAbsentOp(loc, argTy));
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument");

      if (arg.passBy == PassBy::Value) {
        ExtValue argVal = genval(*expr);
        if (!fir::isUnboxedValue(argVal))
          fir::emitFatalError(
              loc, "internal error: passing non trivial value by value");
        caller.placeInput(arg, fir::getBase(argVal));
        continue;
      }

      if (arg.passBy == PassBy::MutableBox) {
        if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
                *expr)) {
          // If expr is NULL(), the mutableBox created must be a deallocated
          // pointer with the dummy argument characteristics (see table 16.5
          // in Fortran 2018 standard).
          // No length parameters are set for the created box because any non
          // deferred type parameters of the dummy will be evaluated on the
          // callee side, and it is illegal to use NULL without a MOLD if any
          // dummy length parameters are assumed.
          mlir::Type boxTy = fir::dyn_cast_ptrEleTy(argTy);
          assert(boxTy && boxTy.isa<fir::BaseBoxType>() &&
                 "must be a fir.box type");
          mlir::Value boxStorage = builder.createTemporary(loc, boxTy);
          mlir::Value nullBox = fir::factory::createUnallocatedBox(
              builder, loc, boxTy, /*nonDeferredParams=*/{});
          builder.create<fir::StoreOp>(loc, nullBox, boxStorage);
          caller.placeInput(arg, boxStorage);
          continue;
        }
        if (fir::isPointerType(argTy) &&
            !Fortran::evaluate::IsObjectPointer(
                *expr, converter.getFoldingContext())) {
          // Passing a non POINTER actual argument to a POINTER dummy argument.
          // Create a pointer of the dummy argument type and assign the actual
          // argument to it.
          mlir::Value irBox =
              builder.createTemporary(loc, fir::unwrapRefType(argTy));
          // Non deferred parameters will be evaluated on the callee side.
          fir::MutableBoxValue pointer(irBox,
                                       /*nonDeferredParams=*/mlir::ValueRange{},
                                       /*mutableProperties=*/{});
          Fortran::lower::associateMutableBox(converter, loc, pointer, *expr,
                                              /*lbounds=*/std::nullopt,
                                              stmtCtx);
          caller.placeInput(arg, irBox);
          continue;
        }
        // Passing a POINTER to a POINTER, or an ALLOCATABLE to an ALLOCATABLE.
        fir::MutableBoxValue mutableBox = genMutableBoxValue(*expr);
        mlir::Value irBox =
            fir::factory::getMutableIRBox(builder, loc, mutableBox);
        caller.placeInput(arg, irBox);
        if (arg.mayBeModifiedByCall())
          mutableModifiedByCall.emplace_back(std::move(mutableBox));
        if (fir::isAllocatableType(argTy) && arg.isIntentOut() &&
            Fortran::semantics::IsBindCProcedure(*procRef.proc().GetSymbol())) {
          if (mutableBox.isDerived() || mutableBox.isPolymorphic() ||
              mutableBox.isUnlimitedPolymorphic()) {
            mlir::Value isAlloc = fir::factory::genIsAllocatedOrAssociatedTest(
                builder, loc, mutableBox);
            builder.genIfThen(loc, isAlloc)
                .genThen([&]() {
                  Fortran::lower::genDeallocateBox(converter, mutableBox, loc);
                })
                .end();
          } else {
            Fortran::lower::genDeallocateBox(converter, mutableBox, loc);
          }
        }
        continue;
      }
      if (arg.passBy == PassBy::BaseAddress || arg.passBy == PassBy::BoxChar ||
          arg.passBy == PassBy::BaseAddressValueAttribute ||
          arg.passBy == PassBy::CharBoxValueAttribute) {
        const bool byValue = arg.passBy == PassBy::BaseAddressValueAttribute ||
                             arg.passBy == PassBy::CharBoxValueAttribute;
        ExtValue argAddr =
            prepareActualToBaseAddressLike(*expr, arg, copyOutPairs, byValue)
                .first;
        if (arg.passBy == PassBy::BaseAddress ||
            arg.passBy == PassBy::BaseAddressValueAttribute) {
          caller.placeInput(arg, fir::getBase(argAddr));
        } else {
          assert(arg.passBy == PassBy::BoxChar ||
                 arg.passBy == PassBy::CharBoxValueAttribute);
          auto helper = fir::factory::CharacterExprHelper{builder, loc};
          auto boxChar = argAddr.match(
              [&](const fir::CharBoxValue &x) -> mlir::Value {
                // If a character procedure was passed instead, handle the
                // mismatch.
                auto funcTy =
                    x.getAddr().getType().dyn_cast<mlir::FunctionType>();
                if (funcTy && funcTy.getNumResults() == 1 &&
                    funcTy.getResult(0).isa<fir::BoxCharType>()) {
                  auto boxTy = funcTy.getResult(0).cast<fir::BoxCharType>();
                  mlir::Value ref = builder.createConvert(
                      loc, builder.getRefType(boxTy.getEleTy()), x.getAddr());
                  auto len = builder.create<fir::UndefOp>(
                      loc, builder.getCharacterLengthType());
                  return builder.create<fir::EmboxCharOp>(loc, boxTy, ref, len);
                }
                return helper.createEmbox(x);
              },
              [&](const fir::CharArrayBoxValue &x) {
                return helper.createEmbox(x);
              },
              [&](const auto &x) -> mlir::Value {
                // Fortran allows an actual argument of a completely different
                // type to be passed to a procedure expecting a CHARACTER in the
                // dummy argument position. When this happens, the data pointer
                // argument is simply assumed to point to CHARACTER data and the
                // LEN argument used is garbage. Simulate this behavior by
                // free-casting the base address to be a !fir.char reference and
                // setting the LEN argument to undefined. What could go wrong?
                auto dataPtr = fir::getBase(x);
                assert(!dataPtr.getType().template isa<fir::BoxType>());
                return builder.convertWithSemantics(
                    loc, argTy, dataPtr,
                    /*allowCharacterConversion=*/true);
              });
          caller.placeInput(arg, boxChar);
        }
      } else if (arg.passBy == PassBy::Box) {
        if (arg.mustBeMadeContiguous() &&
            !Fortran::evaluate::IsSimplyContiguous(
                *expr, converter.getFoldingContext())) {
          // If the expression is a PDT, or a polymorphic entity, or an assumed
          // rank, it cannot currently be safely handled by
          // prepareActualToBaseAddressLike that is intended to prepare
          // arguments that can be passed as simple base address.
          if (auto dynamicType = expr->GetType())
            if (dynamicType->IsPolymorphic())
              TODO(loc, "passing a polymorphic entity to an OPTIONAL "
                        "CONTIGUOUS argument");
          if (fir::isRecordWithTypeParameters(
                  fir::unwrapSequenceType(fir::unwrapPassByRefType(argTy))))
            TODO(loc, "passing to an OPTIONAL CONTIGUOUS derived type argument "
                      "with length parameters");
          if (Fortran::evaluate::IsAssumedRank(*expr))
            TODO(loc, "passing an assumed rank entity to an OPTIONAL "
                      "CONTIGUOUS argument");
          // Assumed shape VALUE are currently TODO in the call interface
          // lowering.
          const bool byValue = false;
          auto [argAddr, isPresentValue] =
              prepareActualToBaseAddressLike(*expr, arg, copyOutPairs, byValue);
          mlir::Value box = builder.createBox(loc, argAddr);
          if (isPresentValue) {
            mlir::Value convertedBox = builder.createConvert(loc, argTy, box);
            auto absent = builder.create<fir::AbsentOp>(loc, argTy);
            caller.placeInput(arg,
                              builder.create<mlir::arith::SelectOp>(
                                  loc, *isPresentValue, convertedBox, absent));
          } else {
            caller.placeInput(arg, builder.createBox(loc, argAddr));
          }

        } else if (arg.isOptional() &&
                   Fortran::evaluate::IsAllocatableOrPointerObject(
                       *expr, converter.getFoldingContext())) {
          // Before lowering to an address, handle the allocatable/pointer
          // actual argument to optional fir.box dummy. It is legal to pass
          // unallocated/disassociated entity to an optional. In this case, an
          // absent fir.box must be created instead of a fir.box with a null
          // value (Fortran 2018 15.5.2.12 point 1).
          //
          // Note that passing an absent allocatable to a non-allocatable
          // optional dummy argument is illegal (15.5.2.12 point 3 (8)). So
          // nothing has to be done to generate an absent argument in this case,
          // and it is OK to unconditionally read the mutable box here.
          fir::MutableBoxValue mutableBox = genMutableBoxValue(*expr);
          mlir::Value isAllocated =
              fir::factory::genIsAllocatedOrAssociatedTest(builder, loc,
                                                           mutableBox);
          auto absent = builder.create<fir::AbsentOp>(loc, argTy);
          /// For now, assume it is not OK to pass the allocatable/pointer
          /// descriptor to a non pointer/allocatable dummy. That is a strict
          /// interpretation of 18.3.6 point 4 that stipulates the descriptor
          /// has the dummy attributes in BIND(C) contexts.
          mlir::Value box = builder.createBox(
              loc, fir::factory::genMutableBoxRead(builder, loc, mutableBox));

          // NULL() passed as argument is passed as a !fir.box<none>. Since
          // select op requires the same type for its two argument, convert
          // !fir.box<none> to !fir.class<none> when the argument is
          // polymorphic.
          if (fir::isBoxNone(box.getType()) && fir::isPolymorphicType(argTy)) {
            box = builder.createConvert(
                loc,
                fir::ClassType::get(mlir::NoneType::get(builder.getContext())),
                box);
          } else if (box.getType().isa<fir::BoxType>() &&
                     fir::isPolymorphicType(argTy)) {
            box = builder.create<fir::ReboxOp>(loc, argTy, box, mlir::Value{},
                                               /*slice=*/mlir::Value{});
          }

          // Need the box types to be exactly similar for the selectOp.
          mlir::Value convertedBox = builder.createConvert(loc, argTy, box);
          caller.placeInput(arg, builder.create<mlir::arith::SelectOp>(
                                     loc, isAllocated, convertedBox, absent));
        } else {
          auto dynamicType = expr->GetType();
          mlir::Value box;

          // Special case when an intrinsic scalar variable is passed to a
          // function expecting an optional unlimited polymorphic dummy
          // argument.
          // The presence test needs to be performed before emboxing otherwise
          // the program will crash.
          if (dynamicType->category() !=
                  Fortran::common::TypeCategory::Derived &&
              expr->Rank() == 0 && fir::isUnlimitedPolymorphicType(argTy) &&
              arg.isOptional()) {
            ExtValue opt = lowerIntrinsicArgumentAsInquired(*expr);
            mlir::Value isPresent = genActualIsPresentTest(builder, loc, opt);
            box =
                builder
                    .genIfOp(loc, {argTy}, isPresent, /*withElseRegion=*/true)
                    .genThen([&]() {
                      auto boxed = builder.createBox(
                          loc, genBoxArg(*expr), fir::isPolymorphicType(argTy));
                      builder.create<fir::ResultOp>(loc, boxed);
                    })
                    .genElse([&]() {
                      auto absent =
                          builder.create<fir::AbsentOp>(loc, argTy).getResult();
                      builder.create<fir::ResultOp>(loc, absent);
                    })
                    .getResults()[0];
          } else {
            // Make sure a variable address is only passed if the expression is
            // actually a variable.
            box = Fortran::evaluate::IsVariable(*expr)
                      ? builder.createBox(loc, genBoxArg(*expr),
                                          fir::isPolymorphicType(argTy),
                                          fir::isAssumedType(argTy))
                      : builder.createBox(getLoc(), genTempExtAddr(*expr),
                                          fir::isPolymorphicType(argTy),
                                          fir::isAssumedType(argTy));
            if (box.getType().isa<fir::BoxType>() &&
                fir::isPolymorphicType(argTy) && !fir::isAssumedType(argTy)) {
              mlir::Type actualTy = argTy;
              if (Fortran::lower::isParentComponent(*expr))
                actualTy = fir::BoxType::get(converter.genType(*expr));
              // Rebox can only be performed on a present argument.
              if (arg.isOptional()) {
                mlir::Value isPresent =
                    genActualIsPresentTest(builder, loc, box);
                box = builder
                          .genIfOp(loc, {actualTy}, isPresent,
                                   /*withElseRegion=*/true)
                          .genThen([&]() {
                            auto rebox =
                                builder
                                    .create<fir::ReboxOp>(
                                        loc, actualTy, box, mlir::Value{},
                                        /*slice=*/mlir::Value{})
                                    .getResult();
                            builder.create<fir::ResultOp>(loc, rebox);
                          })
                          .genElse([&]() {
                            auto absent =
                                builder.create<fir::AbsentOp>(loc, actualTy)
                                    .getResult();
                            builder.create<fir::ResultOp>(loc, absent);
                          })
                          .getResults()[0];
              } else {
                box = builder.create<fir::ReboxOp>(loc, actualTy, box,
                                                   mlir::Value{},
                                                   /*slice=*/mlir::Value{});
              }
            } else if (Fortran::lower::isParentComponent(*expr)) {
              fir::ExtendedValue newExv =
                  Fortran::lower::updateBoxForParentComponent(converter, box,
                                                              *expr);
              box = fir::getBase(newExv);
            }
          }
          caller.placeInput(arg, box);
        }
      } else if (arg.passBy == PassBy::AddressAndLength) {
        ExtValue argRef = genExtAddr(*expr);
        caller.placeAddressAndLengthInput(arg, fir::getBase(argRef),
                                          fir::getLen(argRef));
      } else if (arg.passBy == PassBy::CharProcTuple) {
        ExtValue argRef = genExtAddr(*expr);
        mlir::Value tuple = createBoxProcCharTuple(
            converter, argTy, fir::getBase(argRef), fir::getLen(argRef));
        caller.placeInput(arg, tuple);
      } else {
        TODO(loc, "pass by value in non elemental function call");
      }
    }

    ExtValue result = Fortran::lower::genCallOpAndResult(
        loc, converter, symMap, stmtCtx, caller, callSiteType, resultType);

    // Sync pointers and allocatables that may have been modified during the
    // call.
    for (const auto &mutableBox : mutableModifiedByCall)
      fir::factory::syncMutableBoxFromIRBox(builder, loc, mutableBox);
    // Handle case where result was passed as argument

    // Copy-out temps that were created for non contiguous variable arguments if
    // needed.
    for (const auto &copyOutPair : copyOutPairs)
      genCopyOut(copyOutPair);

    return result;
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    ExtValue result = genFunctionRef(funcRef);
    if (result.rank() == 0 && fir::isa_ref_type(fir::getBase(result).getType()))
      return genLoad(result);
    return result;
  }

  ExtValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    std::optional<mlir::Type> resTy;
    if (procRef.hasAlternateReturns())
      resTy = builder.getIndexType();
    return genProcedureRef(procRef, resTy);
  }

  template <typename A>
  bool isScalar(const A &x) {
    return x.Rank() == 0;
  }

  /// Helper to detect Transformational function reference.
  template <typename T>
  bool isTransformationalRef(const T &) {
    return false;
  }
  template <typename T>
  bool isTransformationalRef(const Fortran::evaluate::FunctionRef<T> &funcRef) {
    return !funcRef.IsElemental() && funcRef.Rank();
  }
  template <typename T>
  bool isTransformationalRef(Fortran::evaluate::Expr<T> expr) {
    return std::visit([&](const auto &e) { return isTransformationalRef(e); },
                      expr.u);
  }

  template <typename A>
  ExtValue asArray(const A &x) {
    return Fortran::lower::createSomeArrayTempValue(converter, toEvExpr(x),
                                                    symMap, stmtCtx);
  }

  /// Lower an array value as an argument. This argument can be passed as a box
  /// value, so it may be possible to avoid making a temporary.
  template <typename A>
  ExtValue asArrayArg(const Fortran::evaluate::Expr<A> &x) {
    return std::visit([&](const auto &e) { return asArrayArg(e, x); }, x.u);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const Fortran::evaluate::Expr<A> &x, const B &y) {
    return std::visit([&](const auto &e) { return asArrayArg(e, y); }, x.u);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const Fortran::evaluate::Designator<A> &, const B &x) {
    // Designator is being passed as an argument to a procedure. Lower the
    // expression to a boxed value.
    auto someExpr = toEvExpr(x);
    return Fortran::lower::createBoxValue(getLoc(), converter, someExpr, symMap,
                                          stmtCtx);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const A &, const B &x) {
    // If the expression to pass as an argument is not a designator, then create
    // an array temp.
    return asArray(x);
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Expr<A> &x) {
    // Whole array symbols or components, and results of transformational
    // functions already have a storage and the scalar expression lowering path
    // is used to not create a new temporary storage.
    if (isScalar(x) ||
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(x) ||
        (isTransformationalRef(x) && !isOptimizableTranspose(x, converter)))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
    if (useBoxArg)
      return asArrayArg(x);
    return asArray(x);
  }
  template <typename A>
  ExtValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x) ||
        inInitializer)
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    return asArray(x);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Logical, KIND>> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  using RefSet =
      std::tuple<Fortran::evaluate::ComplexPart, Fortran::evaluate::Substring,
                 Fortran::evaluate::DataRef, Fortran::evaluate::Component,
                 Fortran::evaluate::ArrayRef, Fortran::evaluate::CoarrayRef,
                 Fortran::semantics::SymbolRef>;
  template <typename A>
  static constexpr bool inRefSet = Fortran::common::HasMember<A, RefSet>;

  template <typename A, typename = std::enable_if_t<inRefSet<A>>>
  ExtValue genref(const A &a) {
    return gen(a);
  }
  template <typename A>
  ExtValue genref(const A &a) {
    if (inInitializer) {
      // Initialization expressions can never allocate memory.
      return genval(a);
    }
    mlir::Type storageType = converter.genType(toEvExpr(a));
    return placeScalarValueInMemory(builder, getLoc(), genval(a), storageType);
  }

  template <typename A, template <typename> typename T,
            typename B = std::decay_t<T<A>>,
            std::enable_if_t<
                std::is_same_v<B, Fortran::evaluate::Expr<A>> ||
                    std::is_same_v<B, Fortran::evaluate::Designator<A>> ||
                    std::is_same_v<B, Fortran::evaluate::FunctionRef<A>>,
                bool> = true>
  ExtValue genref(const T<A> &x) {
    return gen(x);
  }

private:
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  bool inInitializer = false;
  bool useBoxArg = false; // expression lowered as argument
};
} // namespace

// Helper for changing the semantics in a given context. Preserves the current
// semantics which is resumed when the "push" goes out of scope.
#define PushSemantics(PushVal)                                                 \
  [[maybe_unused]] auto pushSemanticsLocalVariable##__LINE__ =                 \
      Fortran::common::ScopedSet(semant, PushVal);

static bool isAdjustedArrayElementType(mlir::Type t) {
  return fir::isa_char(t) || fir::isa_derived(t) || t.isa<fir::SequenceType>();
}
static bool elementTypeWasAdjusted(mlir::Type t) {
  if (auto ty = t.dyn_cast<fir::ReferenceType>())
    return isAdjustedArrayElementType(ty.getEleTy());
  return false;
}
static mlir::Type adjustedArrayElementType(mlir::Type t) {
  return isAdjustedArrayElementType(t) ? fir::ReferenceType::get(t) : t;
}

/// Helper to generate calls to scalar user defined assignment procedures.
static void genScalarUserDefinedAssignmentCall(fir::FirOpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::func::FuncOp func,
                                               const fir::ExtendedValue &lhs,
                                               const fir::ExtendedValue &rhs) {
  auto prepareUserDefinedArg =
      [](fir::FirOpBuilder &builder, mlir::Location loc,
         const fir::ExtendedValue &value, mlir::Type argType) -> mlir::Value {
    if (argType.isa<fir::BoxCharType>()) {
      const fir::CharBoxValue *charBox = value.getCharBox();
      assert(charBox && "argument type mismatch in elemental user assignment");
      return fir::factory::CharacterExprHelper{builder, loc}.createEmbox(
          *charBox);
    }
    if (argType.isa<fir::BaseBoxType>()) {
      mlir::Value box =
          builder.createBox(loc, value, argType.isa<fir::ClassType>());
      return builder.createConvert(loc, argType, box);
    }
    // Simple pass by address.
    mlir::Type argBaseType = fir::unwrapRefType(argType);
    assert(!fir::hasDynamicSize(argBaseType));
    mlir::Value from = fir::getBase(value);
    if (argBaseType != fir::unwrapRefType(from.getType())) {
      // With logicals, it is possible that from is i1 here.
      if (fir::isa_ref_type(from.getType()))
        from = builder.create<fir::LoadOp>(loc, from);
      from = builder.createConvert(loc, argBaseType, from);
    }
    if (!fir::isa_ref_type(from.getType())) {
      mlir::Value temp = builder.createTemporary(loc, argBaseType);
      builder.create<fir::StoreOp>(loc, from, temp);
      from = temp;
    }
    return builder.createConvert(loc, argType, from);
  };
  assert(func.getNumArguments() == 2);
  mlir::Type lhsType = func.getFunctionType().getInput(0);
  mlir::Type rhsType = func.getFunctionType().getInput(1);
  mlir::Value lhsArg = prepareUserDefinedArg(builder, loc, lhs, lhsType);
  mlir::Value rhsArg = prepareUserDefinedArg(builder, loc, rhs, rhsType);
  builder.create<fir::CallOp>(loc, func, mlir::ValueRange{lhsArg, rhsArg});
}

/// Convert the result of a fir.array_modify to an ExtendedValue given the
/// related fir.array_load.
static fir::ExtendedValue arrayModifyToExv(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           fir::ArrayLoadOp load,
                                           mlir::Value elementAddr) {
  mlir::Type eleTy = fir::unwrapPassByRefType(elementAddr.getType());
  if (fir::isa_char(eleTy)) {
    auto len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
        load.getMemref());
    if (!len) {
      assert(load.getTypeparams().size() == 1 &&
             "length must be in array_load");
      len = load.getTypeparams()[0];
    }
    return fir::CharBoxValue{elementAddr, len};
  }
  return elementAddr;
}

//===----------------------------------------------------------------------===//
//
// Lowering of scalar expressions in an explicit iteration space context.
//
//===----------------------------------------------------------------------===//

// Shared code for creating a copy of a derived type element. This function is
// called from a continuation.
inline static fir::ArrayAmendOp
createDerivedArrayAmend(mlir::Location loc, fir::ArrayLoadOp destLoad,
                        fir::FirOpBuilder &builder, fir::ArrayAccessOp destAcc,
                        const fir::ExtendedValue &elementExv, mlir::Type eleTy,
                        mlir::Value innerArg) {
  if (destLoad.getTypeparams().empty()) {
    fir::factory::genRecordAssignment(builder, loc, destAcc, elementExv);
  } else {
    auto boxTy = fir::BoxType::get(eleTy);
    auto toBox = builder.create<fir::EmboxOp>(loc, boxTy, destAcc.getResult(),
                                              mlir::Value{}, mlir::Value{},
                                              destLoad.getTypeparams());
    auto fromBox = builder.create<fir::EmboxOp>(
        loc, boxTy, fir::getBase(elementExv), mlir::Value{}, mlir::Value{},
        destLoad.getTypeparams());
    fir::factory::genRecordAssignment(builder, loc, fir::BoxValue(toBox),
                                      fir::BoxValue(fromBox));
  }
  return builder.create<fir::ArrayAmendOp>(loc, innerArg.getType(), innerArg,
                                           destAcc);
}

inline static fir::ArrayAmendOp
createCharArrayAmend(mlir::Location loc, fir::FirOpBuilder &builder,
                     fir::ArrayAccessOp dstOp, mlir::Value &dstLen,
                     const fir::ExtendedValue &srcExv, mlir::Value innerArg,
                     llvm::ArrayRef<mlir::Value> bounds) {
  fir::CharBoxValue dstChar(dstOp, dstLen);
  fir::factory::CharacterExprHelper helper{builder, loc};
  if (!bounds.empty()) {
    dstChar = helper.createSubstring(dstChar, bounds);
    fir::factory::genCharacterCopy(fir::getBase(srcExv), fir::getLen(srcExv),
                                   dstChar.getAddr(), dstChar.getLen(), builder,
                                   loc);
    // Update the LEN to the substring's LEN.
    dstLen = dstChar.getLen();
  }
  // For a CHARACTER, we generate the element assignment loops inline.
  helper.createAssign(fir::ExtendedValue{dstChar}, srcExv);
  // Mark this array element as amended.
  mlir::Type ty = innerArg.getType();
  auto amend = builder.create<fir::ArrayAmendOp>(loc, ty, innerArg, dstOp);
  return amend;
}

/// Build an ExtendedValue from a fir.array<?x...?xT> without actually setting
/// the actual extents and lengths. This is only to allow their propagation as
/// ExtendedValue without triggering verifier failures when propagating
/// character/arrays as unboxed values. Only the base of the resulting
/// ExtendedValue should be used, it is undefined to use the length or extents
/// of the extended value returned,
inline static fir::ExtendedValue
convertToArrayBoxValue(mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::Value val, mlir::Value len) {
  mlir::Type ty = fir::unwrapRefType(val.getType());
  mlir::IndexType idxTy = builder.getIndexType();
  auto seqTy = ty.cast<fir::SequenceType>();
  auto undef = builder.create<fir::UndefOp>(loc, idxTy);
  llvm::SmallVector<mlir::Value> extents(seqTy.getDimension(), undef);
  if (fir::isa_char(seqTy.getEleTy()))
    return fir::CharArrayBoxValue(val, len ? len : undef, extents);
  return fir::ArrayBoxValue(val, extents);
}

//===----------------------------------------------------------------------===//
//
// Lowering of array expressions.
//
//===----------------------------------------------------------------------===//

namespace {
class ArrayExprLowering {
  using ExtValue = fir::ExtendedValue;

  /// Structure to keep track of lowered array operands in the
  /// array expression. Useful to later deduce the shape of the
  /// array expression.
  struct ArrayOperand {
    /// Array base (can be a fir.box).
    mlir::Value memref;
    /// ShapeOp, ShapeShiftOp or ShiftOp
    mlir::Value shape;
    /// SliceOp
    mlir::Value slice;
    /// Can this operand be absent ?
    bool mayBeAbsent = false;
  };

  using ImplicitSubscripts = Fortran::lower::details::ImplicitSubscripts;
  using PathComponent = Fortran::lower::PathComponent;

  /// Active iteration space.
  using IterationSpace = Fortran::lower::IterationSpace;
  using IterSpace = const Fortran::lower::IterationSpace &;

  /// Current continuation. Function that will generate IR for a single
  /// iteration of the pending iterative loop structure.
  using CC = Fortran::lower::GenerateElementalArrayFunc;

  /// Projection continuation. Function that will project one iteration space
  /// into another.
  using PC = std::function<IterationSpace(IterSpace)>;
  using ArrayBaseTy =
      std::variant<std::monostate, const Fortran::evaluate::ArrayRef *,
                   const Fortran::evaluate::DataRef *>;
  using ComponentPath = Fortran::lower::ComponentPath;

public:
  //===--------------------------------------------------------------------===//
  // Regular array assignment
  //===--------------------------------------------------------------------===//

  /// Entry point for array assignments. Both the left-hand and right-hand sides
  /// can either be ExtendedValue or evaluate::Expr.
  template <typename TL, typename TR>
  static void lowerArrayAssignment(Fortran::lower::AbstractConverter &converter,
                                   Fortran::lower::SymMap &symMap,
                                   Fortran::lower::StatementContext &stmtCtx,
                                   const TL &lhs, const TR &rhs) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut);
    ael.lowerArrayAssignment(lhs, rhs);
  }

  template <typename TL, typename TR>
  void lowerArrayAssignment(const TL &lhs, const TR &rhs) {
    mlir::Location loc = getLoc();
    /// Here the target subspace is not necessarily contiguous. The ArrayUpdate
    /// continuation is implicitly returned in `ccStoreToDest` and the ArrayLoad
    /// in `destination`.
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    ccStoreToDest = genarr(lhs);
    determineShapeOfDest(lhs);
    semant = ConstituentSemantics::RefTransparent;
    ExtValue exv = lowerArrayExpression(rhs);
    if (explicitSpaceIsActive()) {
      explicitSpace->finalizeContext();
      builder.create<fir::ResultOp>(loc, fir::getBase(exv));
    } else {
      builder.create<fir::ArrayMergeStoreOp>(
          loc, destination, fir::getBase(exv), destination.getMemref(),
          destination.getSlice(), destination.getTypeparams());
    }
  }

  //===--------------------------------------------------------------------===//
  // WHERE array assignment, FORALL assignment, and FORALL+WHERE array
  // assignment
  //===--------------------------------------------------------------------===//

  /// Entry point for array assignment when the iteration space is explicitly
  /// defined (Fortran's FORALL) with or without masks, and/or the implied
  /// iteration space involves masks (Fortran's WHERE). Both contexts (explicit
  /// space and implicit space with masks) may be present.
  static void lowerAnyMaskedArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
      Fortran::lower::ExplicitIterSpace &explicitSpace,
      Fortran::lower::ImplicitIterSpace &implicitSpace) {
    if (explicitSpace.isActive() && lhs.Rank() == 0) {
      // Scalar assignment expression in a FORALL context.
      ArrayExprLowering ael(converter, stmtCtx, symMap,
                            ConstituentSemantics::RefTransparent,
                            &explicitSpace, &implicitSpace);
      ael.lowerScalarAssignment(lhs, rhs);
      return;
    }
    // Array assignment expression in a FORALL and/or WHERE context.
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &explicitSpace,
                          &implicitSpace);
    ael.lowerArrayAssignment(lhs, rhs);
  }

  //===--------------------------------------------------------------------===//
  // Array assignment to array of pointer box values.
  //===--------------------------------------------------------------------===//

  /// Entry point for assignment to pointer in an array of pointers.
  static void lowerArrayOfPointerAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
      Fortran::lower::ExplicitIterSpace &explicitSpace,
      Fortran::lower::ImplicitIterSpace &implicitSpace,
      const llvm::SmallVector<mlir::Value> &lbounds,
      std::optional<llvm::SmallVector<mlir::Value>> ubounds) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &explicitSpace,
                          &implicitSpace);
    ael.lowerArrayOfPointerAssignment(lhs, rhs, lbounds, ubounds);
  }

  /// Scalar pointer assignment in an explicit iteration space.
  ///
  /// Pointers may be bound to targets in a FORALL context. This is a scalar
  /// assignment in the sense there is never an implied iteration space, even if
  /// the pointer is to a target with non-zero rank. Since the pointer
  /// assignment must appear in a FORALL construct, correctness may require that
  /// the array of pointers follow copy-in/copy-out semantics. The pointer
  /// assignment may include a bounds-spec (lower bounds), a bounds-remapping
  /// (lower and upper bounds), or neither.
  void lowerArrayOfPointerAssignment(
      const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
      const llvm::SmallVector<mlir::Value> &lbounds,
      std::optional<llvm::SmallVector<mlir::Value>> ubounds) {
    setPointerAssignmentBounds(lbounds, ubounds);
    if (rhs.Rank() == 0 ||
        (Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(rhs) &&
         Fortran::evaluate::IsAllocatableOrPointerObject(
             rhs, converter.getFoldingContext()))) {
      lowerScalarAssignment(lhs, rhs);
      return;
    }
    TODO(getLoc(),
         "auto boxing of a ranked expression on RHS for pointer assignment");
  }

  //===--------------------------------------------------------------------===//
  // Array assignment to allocatable array
  //===--------------------------------------------------------------------===//

  /// Entry point for assignment to allocatable array.
  static void lowerAllocatableArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
      Fortran::lower::ExplicitIterSpace &explicitSpace,
      Fortran::lower::ImplicitIterSpace &implicitSpace) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &explicitSpace,
                          &implicitSpace);
    ael.lowerAllocatableArrayAssignment(lhs, rhs);
  }

  /// Lower an assignment to allocatable array, where the LHS array
  /// is represented with \p lhs extended value produced in different
  /// branches created in genReallocIfNeeded(). The RHS lowering
  /// is provided via \p rhsCC continuation.
  void lowerAllocatableArrayAssignment(ExtValue lhs, CC rhsCC) {
    mlir::Location loc = getLoc();
    // Check if the initial destShape is null, which means
    // it has not been computed from rhs (e.g. rhs is scalar).
    bool destShapeIsEmpty = destShape.empty();
    // Create ArrayLoad for the mutable box and save it into `destination`.
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    ccStoreToDest = genarr(lhs);
    // destShape is either non-null on entry to this function,
    // or has been just set by lhs lowering.
    assert(!destShape.empty() && "destShape must have been set.");
    // Finish lowering the loop nest.
    assert(destination && "destination must have been set");
    ExtValue exv = lowerArrayExpression(rhsCC, destination.getType());
    if (!explicitSpaceIsActive())
      builder.create<fir::ArrayMergeStoreOp>(
          loc, destination, fir::getBase(exv), destination.getMemref(),
          destination.getSlice(), destination.getTypeparams());
    // destShape may originally be null, if rhs did not define a shape.
    // In this case the destShape is computed from lhs, and we may have
    // multiple different lhs values for different branches created
    // in genReallocIfNeeded(). We cannot reuse destShape computed
    // in different branches, so we have to reset it,
    // so that it is recomputed for the next branch FIR generation.
    if (destShapeIsEmpty)
      destShape.clear();
  }

  /// Assignment to allocatable array.
  ///
  /// The semantics are reverse that of a "regular" array assignment. The rhs
  /// defines the iteration space of the computation and the lhs is
  /// resized/reallocated to fit if necessary.
  void lowerAllocatableArrayAssignment(const Fortran::lower::SomeExpr &lhs,
                                       const Fortran::lower::SomeExpr &rhs) {
    // With assignment to allocatable, we want to lower the rhs first and use
    // its shape to determine if we need to reallocate, etc.
    mlir::Location loc = getLoc();
    // FIXME: If the lhs is in an explicit iteration space, the assignment may
    // be to an array of allocatable arrays rather than a single allocatable
    // array.
    if (explicitSpaceIsActive() && lhs.Rank() > 0)
      TODO(loc, "assignment to whole allocatable array inside FORALL");

    fir::MutableBoxValue mutableBox =
        Fortran::lower::createMutableBox(loc, converter, lhs, symMap);
    if (rhs.Rank() > 0)
      determineShapeOfDest(rhs);
    auto rhsCC = [&]() {
      PushSemantics(ConstituentSemantics::RefTransparent);
      return genarr(rhs);
    }();

    llvm::SmallVector<mlir::Value> lengthParams;
    // Currently no safe way to gather length from rhs (at least for
    // character, it cannot be taken from array_loads since it may be
    // changed by concatenations).
    if ((mutableBox.isCharacter() && !mutableBox.hasNonDeferredLenParams()) ||
        mutableBox.isDerivedWithLenParameters())
      TODO(loc, "gather rhs LEN parameters in assignment to allocatable");

    // The allocatable must take lower bounds from the expr if it is
    // reallocated and the right hand side is not a scalar.
    const bool takeLboundsIfRealloc = rhs.Rank() > 0;
    llvm::SmallVector<mlir::Value> lbounds;
    // When the reallocated LHS takes its lower bounds from the RHS,
    // they will be non default only if the RHS is a whole array
    // variable. Otherwise, lbounds is left empty and default lower bounds
    // will be used.
    if (takeLboundsIfRealloc &&
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(rhs)) {
      assert(arrayOperands.size() == 1 &&
             "lbounds can only come from one array");
      auto lbs = fir::factory::getOrigins(arrayOperands[0].shape);
      lbounds.append(lbs.begin(), lbs.end());
    }
    auto assignToStorage = [&](fir::ExtendedValue newLhs) {
      // The lambda will be called repeatedly by genReallocIfNeeded().
      lowerAllocatableArrayAssignment(newLhs, rhsCC);
    };
    fir::factory::MutableBoxReallocation realloc =
        fir::factory::genReallocIfNeeded(builder, loc, mutableBox, destShape,
                                         lengthParams, assignToStorage);
    if (explicitSpaceIsActive()) {
      explicitSpace->finalizeContext();
      builder.create<fir::ResultOp>(loc, fir::getBase(realloc.newValue));
    }
    fir::factory::finalizeRealloc(builder, loc, mutableBox, lbounds,
                                  takeLboundsIfRealloc, realloc);
  }

  /// Entry point for when an array expression appears in a context where the
  /// result must be boxed. (BoxValue semantics.)
  static ExtValue
  lowerBoxedArrayExpression(Fortran::lower::AbstractConverter &converter,
                            Fortran::lower::SymMap &symMap,
                            Fortran::lower::StatementContext &stmtCtx,
                            const Fortran::lower::SomeExpr &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::BoxValue};
    return ael.lowerBoxedArrayExpr(expr);
  }

  ExtValue lowerBoxedArrayExpr(const Fortran::lower::SomeExpr &exp) {
    PushSemantics(ConstituentSemantics::BoxValue);
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          ExtValue exv = f(IterationSpace{});
          if (fir::getBase(exv).getType().template isa<fir::BaseBoxType>())
            return exv;
          fir::emitFatalError(getLoc(), "array must be emboxed");
        },
        exp.u);
  }

  /// Entry point into lowering an expression with rank. This entry point is for
  /// lowering a rhs expression, for example. (RefTransparent semantics.)
  static ExtValue
  lowerNewArrayExpression(Fortran::lower::AbstractConverter &converter,
                          Fortran::lower::SymMap &symMap,
                          Fortran::lower::StatementContext &stmtCtx,
                          const Fortran::lower::SomeExpr &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap};
    ael.determineShapeOfDest(expr);
    ExtValue loopRes = ael.lowerArrayExpression(expr);
    fir::ArrayLoadOp dest = ael.destination;
    mlir::Value tempRes = dest.getMemref();
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Location loc = converter.getCurrentLocation();
    builder.create<fir::ArrayMergeStoreOp>(loc, dest, fir::getBase(loopRes),
                                           tempRes, dest.getSlice(),
                                           dest.getTypeparams());

    auto arrTy =
        fir::dyn_cast_ptrEleTy(tempRes.getType()).cast<fir::SequenceType>();
    if (auto charTy =
            arrTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      if (fir::characterWithDynamicLen(charTy))
        TODO(loc, "CHARACTER does not have constant LEN");
      mlir::Value len = builder.createIntegerConstant(
          loc, builder.getCharacterLengthType(), charTy.getLen());
      return fir::CharArrayBoxValue(tempRes, len, dest.getExtents());
    }
    return fir::ArrayBoxValue(tempRes, dest.getExtents());
  }

  static void lowerLazyArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::lower::SomeExpr &expr, mlir::Value raggedHeader) {
    ArrayExprLowering ael(converter, stmtCtx, symMap);
    ael.lowerLazyArrayExpression(expr, raggedHeader);
  }

  /// Lower the expression \p expr into a buffer that is created on demand. The
  /// variable containing the pointer to the buffer is \p var and the variable
  /// containing the shape of the buffer is \p shapeBuffer.
  void lowerLazyArrayExpression(const Fortran::lower::SomeExpr &expr,
                                mlir::Value header) {
    mlir::Location loc = getLoc();
    mlir::TupleType hdrTy = fir::factory::getRaggedArrayHeaderType(builder);
    mlir::IntegerType i32Ty = builder.getIntegerType(32);

    // Once the loop extents have been computed, which may require being inside
    // some explicit loops, lazily allocate the expression on the heap. The
    // following continuation creates the buffer as needed.
    ccPrelude = [=](llvm::ArrayRef<mlir::Value> shape) {
      mlir::IntegerType i64Ty = builder.getIntegerType(64);
      mlir::Value byteSize = builder.createIntegerConstant(loc, i64Ty, 1);
      fir::runtime::genRaggedArrayAllocate(
          loc, builder, header, /*asHeaders=*/false, byteSize, shape);
    };

    // Create a dummy array_load before the loop. We're storing to a lazy
    // temporary, so there will be no conflict and no copy-in. TODO: skip this
    // as there isn't any necessity for it.
    ccLoadDest = [=](llvm::ArrayRef<mlir::Value> shape) -> fir::ArrayLoadOp {
      mlir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
      auto var = builder.create<fir::CoordinateOp>(
          loc, builder.getRefType(hdrTy.getType(1)), header, one);
      auto load = builder.create<fir::LoadOp>(loc, var);
      mlir::Type eleTy =
          fir::unwrapSequenceType(fir::unwrapRefType(load.getType()));
      auto seqTy = fir::SequenceType::get(eleTy, shape.size());
      mlir::Value castTo =
          builder.createConvert(loc, fir::HeapType::get(seqTy), load);
      mlir::Value shapeOp = builder.genShape(loc, shape);
      return builder.create<fir::ArrayLoadOp>(
          loc, seqTy, castTo, shapeOp, /*slice=*/mlir::Value{}, std::nullopt);
    };
    // Custom lowering of the element store to deal with the extra indirection
    // to the lazy allocated buffer.
    ccStoreToDest = [=](IterSpace iters) {
      mlir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
      auto var = builder.create<fir::CoordinateOp>(
          loc, builder.getRefType(hdrTy.getType(1)), header, one);
      auto load = builder.create<fir::LoadOp>(loc, var);
      mlir::Type eleTy =
          fir::unwrapSequenceType(fir::unwrapRefType(load.getType()));
      auto seqTy = fir::SequenceType::get(eleTy, iters.iterVec().size());
      auto toTy = fir::HeapType::get(seqTy);
      mlir::Value castTo = builder.createConvert(loc, toTy, load);
      mlir::Value shape = builder.genShape(loc, genIterationShape());
      llvm::SmallVector<mlir::Value> indices = fir::factory::originateIndices(
          loc, builder, castTo.getType(), shape, iters.iterVec());
      auto eleAddr = builder.create<fir::ArrayCoorOp>(
          loc, builder.getRefType(eleTy), castTo, shape,
          /*slice=*/mlir::Value{}, indices, destination.getTypeparams());
      mlir::Value eleVal =
          builder.createConvert(loc, eleTy, iters.getElement());
      builder.create<fir::StoreOp>(loc, eleVal, eleAddr);
      return iters.innerArgument();
    };

    // Lower the array expression now. Clean-up any temps that may have
    // been generated when lowering `expr` right after the lowered value
    // was stored to the ragged array temporary. The local temps will not
    // be needed afterwards.
    stmtCtx.pushScope();
    [[maybe_unused]] ExtValue loopRes = lowerArrayExpression(expr);
    stmtCtx.finalizeAndPop();
    assert(fir::getBase(loopRes));
  }

  static void
  lowerElementalUserAssignment(Fortran::lower::AbstractConverter &converter,
                               Fortran::lower::SymMap &symMap,
                               Fortran::lower::StatementContext &stmtCtx,
                               Fortran::lower::ExplicitIterSpace &explicitSpace,
                               Fortran::lower::ImplicitIterSpace &implicitSpace,
                               const Fortran::evaluate::ProcedureRef &procRef) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CustomCopyInCopyOut,
                          &explicitSpace, &implicitSpace);
    assert(procRef.arguments().size() == 2);
    const auto *lhs = procRef.arguments()[0].value().UnwrapExpr();
    const auto *rhs = procRef.arguments()[1].value().UnwrapExpr();
    assert(lhs && rhs &&
           "user defined assignment arguments must be expressions");
    mlir::func::FuncOp func =
        Fortran::lower::CallerInterface(procRef, converter).getFuncOp();
    ael.lowerElementalUserAssignment(func, *lhs, *rhs);
  }

  void lowerElementalUserAssignment(mlir::func::FuncOp userAssignment,
                                    const Fortran::lower::SomeExpr &lhs,
                                    const Fortran::lower::SomeExpr &rhs) {
    mlir::Location loc = getLoc();
    PushSemantics(ConstituentSemantics::CustomCopyInCopyOut);
    auto genArrayModify = genarr(lhs);
    ccStoreToDest = [=](IterSpace iters) -> ExtValue {
      auto modifiedArray = genArrayModify(iters);
      auto arrayModify = mlir::dyn_cast_or_null<fir::ArrayModifyOp>(
          fir::getBase(modifiedArray).getDefiningOp());
      assert(arrayModify && "must be created by ArrayModifyOp");
      fir::ExtendedValue lhs =
          arrayModifyToExv(builder, loc, destination, arrayModify.getResult(0));
      genScalarUserDefinedAssignmentCall(builder, loc, userAssignment, lhs,
                                         iters.elementExv());
      return modifiedArray;
    };
    determineShapeOfDest(lhs);
    semant = ConstituentSemantics::RefTransparent;
    auto exv = lowerArrayExpression(rhs);
    if (explicitSpaceIsActive()) {
      explicitSpace->finalizeContext();
      builder.create<fir::ResultOp>(loc, fir::getBase(exv));
    } else {
      builder.create<fir::ArrayMergeStoreOp>(
          loc, destination, fir::getBase(exv), destination.getMemref(),
          destination.getSlice(), destination.getTypeparams());
    }
  }

  /// Lower an elemental subroutine call with at least one array argument.
  /// An elemental subroutine is an exception and does not have copy-in/copy-out
  /// semantics. See 15.8.3.
  /// Do NOT use this for user defined assignments.
  static void
  lowerElementalSubroutine(Fortran::lower::AbstractConverter &converter,
                           Fortran::lower::SymMap &symMap,
                           Fortran::lower::StatementContext &stmtCtx,
                           const Fortran::lower::SomeExpr &call) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::RefTransparent);
    ael.lowerElementalSubroutine(call);
  }

  static const std::optional<Fortran::evaluate::ActualArgument>
  extractPassedArgFromProcRef(const Fortran::evaluate::ProcedureRef &procRef,
                              Fortran::lower::AbstractConverter &converter) {
    // First look for passed object in actual arguments.
    for (const std::optional<Fortran::evaluate::ActualArgument> &arg :
         procRef.arguments())
      if (arg && arg->isPassedObject())
        return arg;

    // If passed object is not found by here, it means the call was fully
    // resolved to the correct procedure. Look for the pass object in the
    // dummy arguments. Pick the first polymorphic one.
    Fortran::lower::CallerInterface caller(procRef, converter);
    unsigned idx = 0;
    for (const auto &arg : caller.characterize().dummyArguments) {
      if (const auto *dummy =
              std::get_if<Fortran::evaluate::characteristics::DummyDataObject>(
                  &arg.u))
        if (dummy->type.type().IsPolymorphic())
          return procRef.arguments()[idx];
      ++idx;
    }
    return std::nullopt;
  }

  // TODO: See the comment in genarr(const Fortran::lower::Parentheses<T>&).
  // This is skipping generation of copy-in/copy-out code for analysis that is
  // required when arguments are in parentheses.
  void lowerElementalSubroutine(const Fortran::lower::SomeExpr &call) {
    if (const auto *procRef =
            std::get_if<Fortran::evaluate::ProcedureRef>(&call.u))
      setLoweredProcRef(procRef);
    auto f = genarr(call);
    llvm::SmallVector<mlir::Value> shape = genIterationShape();
    auto [iterSpace, insPt] = genImplicitLoops(shape, /*innerArg=*/{});
    f(iterSpace);
    finalizeElementCtx();
    builder.restoreInsertionPoint(insPt);
  }

  ExtValue lowerScalarAssignment(const Fortran::lower::SomeExpr &lhs,
                                 const Fortran::lower::SomeExpr &rhs) {
    PushSemantics(ConstituentSemantics::RefTransparent);
    // 1) Lower the rhs expression with array_fetch op(s).
    IterationSpace iters;
    iters.setElement(genarr(rhs)(iters));
    // 2) Lower the lhs expression to an array_update.
    semant = ConstituentSemantics::ProjectedCopyInCopyOut;
    auto lexv = genarr(lhs)(iters);
    // 3) Finalize the inner context.
    explicitSpace->finalizeContext();
    // 4) Thread the array value updated forward. Note: the lhs might be
    // ill-formed (performing scalar assignment in an array context),
    // in which case there is no array to thread.
    auto loc = getLoc();
    auto createResult = [&](auto op) {
      mlir::Value oldInnerArg = op.getSequence();
      std::size_t offset = explicitSpace->argPosition(oldInnerArg);
      explicitSpace->setInnerArg(offset, fir::getBase(lexv));
      finalizeElementCtx();
      builder.create<fir::ResultOp>(loc, fir::getBase(lexv));
    };
    if (mlir::Operation *defOp = fir::getBase(lexv).getDefiningOp()) {
      llvm::TypeSwitch<mlir::Operation *>(defOp)
          .Case([&](fir::ArrayUpdateOp op) { createResult(op); })
          .Case([&](fir::ArrayAmendOp op) { createResult(op); })
          .Case([&](fir::ArrayModifyOp op) { createResult(op); })
          .Default([&](mlir::Operation *) { finalizeElementCtx(); });
    } else {
      // `lhs` isn't from a `fir.array_load`, so there is no array modifications
      // to thread through the iteration space.
      finalizeElementCtx();
    }
    return lexv;
  }

  static ExtValue lowerScalarUserAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      Fortran::lower::ExplicitIterSpace &explicitIterSpace,
      mlir::func::FuncOp userAssignmentFunction,
      const Fortran::lower::SomeExpr &lhs,
      const Fortran::lower::SomeExpr &rhs) {
    Fortran::lower::ImplicitIterSpace implicit;
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::RefTransparent,
                          &explicitIterSpace, &implicit);
    return ael.lowerScalarUserAssignment(userAssignmentFunction, lhs, rhs);
  }

  ExtValue lowerScalarUserAssignment(mlir::func::FuncOp userAssignment,
                                     const Fortran::lower::SomeExpr &lhs,
                                     const Fortran::lower::SomeExpr &rhs) {
    mlir::Location loc = getLoc();
    if (rhs.Rank() > 0)
      TODO(loc, "user-defined elemental assigment from expression with rank");
    // 1) Lower the rhs expression with array_fetch op(s).
    IterationSpace iters;
    iters.setElement(genarr(rhs)(iters));
    fir::ExtendedValue elementalExv = iters.elementExv();
    // 2) Lower the lhs expression to an array_modify.
    semant = ConstituentSemantics::CustomCopyInCopyOut;
    auto lexv = genarr(lhs)(iters);
    bool isIllFormedLHS = false;
    // 3) Insert the call
    if (auto modifyOp = mlir::dyn_cast<fir::ArrayModifyOp>(
            fir::getBase(lexv).getDefiningOp())) {
      mlir::Value oldInnerArg = modifyOp.getSequence();
      std::size_t offset = explicitSpace->argPosition(oldInnerArg);
      explicitSpace->setInnerArg(offset, fir::getBase(lexv));
      auto lhsLoad = explicitSpace->getLhsLoad(0);
      assert(lhsLoad.has_value());
      fir::ExtendedValue exv =
          arrayModifyToExv(builder, loc, *lhsLoad, modifyOp.getResult(0));
      genScalarUserDefinedAssignmentCall(builder, loc, userAssignment, exv,
                                         elementalExv);
    } else {
      // LHS is ill formed, it is a scalar with no references to FORALL
      // subscripts, so there is actually no array assignment here. The user
      // code is probably bad, but still insert user assignment call since it
      // was not rejected by semantics (a warning was emitted).
      isIllFormedLHS = true;
      genScalarUserDefinedAssignmentCall(builder, getLoc(), userAssignment,
                                         lexv, elementalExv);
    }
    // 4) Finalize the inner context.
    explicitSpace->finalizeContext();
    // 5). Thread the array value updated forward.
    if (!isIllFormedLHS) {
      finalizeElementCtx();
      builder.create<fir::ResultOp>(getLoc(), fir::getBase(lexv));
    }
    return lexv;
  }

private:
  void determineShapeOfDest(const fir::ExtendedValue &lhs) {
    destShape = fir::factory::getExtents(getLoc(), builder, lhs);
  }

  void determineShapeOfDest(const Fortran::lower::SomeExpr &lhs) {
    if (!destShape.empty())
      return;
    if (explicitSpaceIsActive() && determineShapeWithSlice(lhs))
      return;
    mlir::Type idxTy = builder.getIndexType();
    mlir::Location loc = getLoc();
    if (std::optional<Fortran::evaluate::ConstantSubscripts> constantShape =
            Fortran::evaluate::GetConstantExtents(converter.getFoldingContext(),
                                                  lhs))
      for (Fortran::common::ConstantSubscript extent : *constantShape)
        destShape.push_back(builder.createIntegerConstant(loc, idxTy, extent));
  }

  bool genShapeFromDataRef(const Fortran::semantics::Symbol &x) {
    return false;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::CoarrayRef &) {
    TODO(getLoc(), "coarray ref");
    return false;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::Component &x) {
    return x.base().Rank() > 0 ? genShapeFromDataRef(x.base()) : false;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::ArrayRef &x) {
    if (x.Rank() == 0)
      return false;
    if (x.base().Rank() > 0)
      if (genShapeFromDataRef(x.base()))
        return true;
    // x has rank and x.base did not produce a shape.
    ExtValue exv = x.base().IsSymbol() ? asScalarRef(getFirstSym(x.base()))
                                       : asScalarRef(x.base().GetComponent());
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> definedShape =
        fir::factory::getExtents(loc, builder, exv);
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    for (auto ss : llvm::enumerate(x.subscript())) {
      std::visit(Fortran::common::visitors{
                     [&](const Fortran::evaluate::Triplet &trip) {
                       // For a subscript of triple notation, we compute the
                       // range of this dimension of the iteration space.
                       auto lo = [&]() {
                         if (auto optLo = trip.lower())
                           return fir::getBase(asScalar(*optLo));
                         return getLBound(exv, ss.index(), one);
                       }();
                       auto hi = [&]() {
                         if (auto optHi = trip.upper())
                           return fir::getBase(asScalar(*optHi));
                         return getUBound(exv, ss.index(), one);
                       }();
                       auto step = builder.createConvert(
                           loc, idxTy, fir::getBase(asScalar(trip.stride())));
                       auto extent = builder.genExtentFromTriplet(loc, lo, hi,
                                                                  step, idxTy);
                       destShape.push_back(extent);
                     },
                     [&](auto) {}},
                 ss.value().u);
    }
    return true;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::NamedEntity &x) {
    if (x.IsSymbol())
      return genShapeFromDataRef(getFirstSym(x));
    return genShapeFromDataRef(x.GetComponent());
  }
  bool genShapeFromDataRef(const Fortran::evaluate::DataRef &x) {
    return std::visit([&](const auto &v) { return genShapeFromDataRef(v); },
                      x.u);
  }

  /// When in an explicit space, the ranked component must be evaluated to
  /// determine the actual number of iterations when slicing triples are
  /// present. Lower these expressions here.
  bool determineShapeWithSlice(const Fortran::lower::SomeExpr &lhs) {
    LLVM_DEBUG(Fortran::lower::DumpEvaluateExpr::dump(
        llvm::dbgs() << "determine shape of:\n", lhs));
    // FIXME: We may not want to use ExtractDataRef here since it doesn't deal
    // with substrings, etc.
    std::optional<Fortran::evaluate::DataRef> dref =
        Fortran::evaluate::ExtractDataRef(lhs);
    return dref.has_value() ? genShapeFromDataRef(*dref) : false;
  }

  /// CHARACTER and derived type elements are treated as memory references. The
  /// numeric types are treated as values.
  static mlir::Type adjustedArraySubtype(mlir::Type ty,
                                         mlir::ValueRange indices) {
    mlir::Type pathTy = fir::applyPathToType(ty, indices);
    assert(pathTy && "indices failed to apply to type");
    return adjustedArrayElementType(pathTy);
  }

  /// Lower rhs of an array expression.
  ExtValue lowerArrayExpression(const Fortran::lower::SomeExpr &exp) {
    mlir::Type resTy = converter.genType(exp);

    if (fir::isPolymorphicType(resTy) &&
        Fortran::evaluate::HasVectorSubscript(exp))
      TODO(getLoc(),
           "polymorphic array expression lowering with vector subscript");

    return std::visit(
        [&](const auto &e) { return lowerArrayExpression(genarr(e), resTy); },
        exp.u);
  }
  ExtValue lowerArrayExpression(const ExtValue &exv) {
    assert(!explicitSpace);
    mlir::Type resTy = fir::unwrapPassByRefType(fir::getBase(exv).getType());
    return lowerArrayExpression(genarr(exv), resTy);
  }

  void populateBounds(llvm::SmallVectorImpl<mlir::Value> &bounds,
                      const Fortran::evaluate::Substring *substring) {
    if (!substring)
      return;
    bounds.push_back(fir::getBase(asScalar(substring->lower())));
    if (auto upper = substring->upper())
      bounds.push_back(fir::getBase(asScalar(*upper)));
  }

  /// Convert the original value, \p origVal, to type \p eleTy. When in a
  /// pointer assignment context, generate an appropriate `fir.rebox` for
  /// dealing with any bounds parameters on the pointer assignment.
  mlir::Value convertElementForUpdate(mlir::Location loc, mlir::Type eleTy,
                                      mlir::Value origVal) {
    if (auto origEleTy = fir::dyn_cast_ptrEleTy(origVal.getType()))
      if (origEleTy.isa<fir::BaseBoxType>()) {
        // If origVal is a box variable, load it so it is in the value domain.
        origVal = builder.create<fir::LoadOp>(loc, origVal);
      }
    if (origVal.getType().isa<fir::BoxType>() && !eleTy.isa<fir::BoxType>()) {
      if (isPointerAssignment())
        TODO(loc, "lhs of pointer assignment returned unexpected value");
      TODO(loc, "invalid box conversion in elemental computation");
    }
    if (isPointerAssignment() && eleTy.isa<fir::BoxType>() &&
        !origVal.getType().isa<fir::BoxType>()) {
      // This is a pointer assignment and the rhs is a raw reference to a TARGET
      // in memory. Embox the reference so it can be stored to the boxed
      // POINTER variable.
      assert(fir::isa_ref_type(origVal.getType()));
      if (auto eleTy = fir::dyn_cast_ptrEleTy(origVal.getType());
          fir::hasDynamicSize(eleTy))
        TODO(loc, "TARGET of pointer assignment with runtime size/shape");
      auto memrefTy = fir::boxMemRefType(eleTy.cast<fir::BoxType>());
      auto castTo = builder.createConvert(loc, memrefTy, origVal);
      origVal = builder.create<fir::EmboxOp>(loc, eleTy, castTo);
    }
    mlir::Value val = builder.createConvert(loc, eleTy, origVal);
    if (isBoundsSpec()) {
      assert(lbounds.has_value());
      auto lbs = *lbounds;
      if (lbs.size() > 0) {
        // Rebox the value with user-specified shift.
        auto shiftTy = fir::ShiftType::get(eleTy.getContext(), lbs.size());
        mlir::Value shiftOp = builder.create<fir::ShiftOp>(loc, shiftTy, lbs);
        val = builder.create<fir::ReboxOp>(loc, eleTy, val, shiftOp,
                                           mlir::Value{});
      }
    } else if (isBoundsRemap()) {
      assert(lbounds.has_value());
      auto lbs = *lbounds;
      if (lbs.size() > 0) {
        // Rebox the value with user-specified shift and shape.
        assert(ubounds.has_value());
        auto shapeShiftArgs = flatZip(lbs, *ubounds);
        auto shapeTy = fir::ShapeShiftType::get(eleTy.getContext(), lbs.size());
        mlir::Value shapeShift =
            builder.create<fir::ShapeShiftOp>(loc, shapeTy, shapeShiftArgs);
        val = builder.create<fir::ReboxOp>(loc, eleTy, val, shapeShift,
                                           mlir::Value{});
      }
    }
    return val;
  }

  /// Default store to destination implementation.
  /// This implements the default case, which is to assign the value in
  /// `iters.element` into the destination array, `iters.innerArgument`. Handles
  /// by value and by reference assignment.
  CC defaultStoreToDestination(const Fortran::evaluate::Substring *substring) {
    return [=](IterSpace iterSpace) -> ExtValue {
      mlir::Location loc = getLoc();
      mlir::Value innerArg = iterSpace.innerArgument();
      fir::ExtendedValue exv = iterSpace.elementExv();
      mlir::Type arrTy = innerArg.getType();
      mlir::Type eleTy = fir::applyPathToType(arrTy, iterSpace.iterVec());
      if (isAdjustedArrayElementType(eleTy)) {
        // The elemental update is in the memref domain. Under this semantics,
        // we must always copy the computed new element from its location in
        // memory into the destination array.
        mlir::Type resRefTy = builder.getRefType(eleTy);
        // Get a reference to the array element to be amended.
        auto arrayOp = builder.create<fir::ArrayAccessOp>(
            loc, resRefTy, innerArg, iterSpace.iterVec(),
            fir::factory::getTypeParams(loc, builder, destination));
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          llvm::SmallVector<mlir::Value> substringBounds;
          populateBounds(substringBounds, substring);
          mlir::Value dstLen = fir::factory::genLenOfCharacter(
              builder, loc, destination, iterSpace.iterVec(), substringBounds);
          fir::ArrayAmendOp amend = createCharArrayAmend(
              loc, builder, arrayOp, dstLen, exv, innerArg, substringBounds);
          return abstractArrayExtValue(amend, dstLen);
        }
        if (fir::isa_derived(eleTy)) {
          fir::ArrayAmendOp amend = createDerivedArrayAmend(
              loc, destination, builder, arrayOp, exv, eleTy, innerArg);
          return abstractArrayExtValue(amend /*FIXME: typeparams?*/);
        }
        assert(eleTy.isa<fir::SequenceType>() && "must be an array");
        TODO(loc, "array (as element) assignment");
      }
      // By value semantics. The element is being assigned by value.
      auto ele = convertElementForUpdate(loc, eleTy, fir::getBase(exv));
      auto update = builder.create<fir::ArrayUpdateOp>(
          loc, arrTy, innerArg, ele, iterSpace.iterVec(),
          destination.getTypeparams());
      return abstractArrayExtValue(update);
    };
  }

  /// For an elemental array expression.
  ///   1. Lower the scalars and array loads.
  ///   2. Create the iteration space.
  ///   3. Create the element-by-element computation in the loop.
  ///   4. Return the resulting array value.
  /// If no destination was set in the array context, a temporary of
  /// \p resultTy will be created to hold the evaluated expression.
  /// Otherwise, \p resultTy is ignored and the expression is evaluated
  /// in the destination. \p f is a continuation built from an
  /// evaluate::Expr or an ExtendedValue.
  ExtValue lowerArrayExpression(CC f, mlir::Type resultTy) {
    mlir::Location loc = getLoc();
    auto [iterSpace, insPt] = genIterSpace(resultTy);
    auto exv = f(iterSpace);
    iterSpace.setElement(std::move(exv));
    auto lambda = ccStoreToDest
                      ? *ccStoreToDest
                      : defaultStoreToDestination(/*substring=*/nullptr);
    mlir::Value updVal = fir::getBase(lambda(iterSpace));
    finalizeElementCtx();
    builder.create<fir::ResultOp>(loc, updVal);
    builder.restoreInsertionPoint(insPt);
    return abstractArrayExtValue(iterSpace.outerResult());
  }

  /// Compute the shape of a slice.
  llvm::SmallVector<mlir::Value> computeSliceShape(mlir::Value slice) {
    llvm::SmallVector<mlir::Value> slicedShape;
    auto slOp = mlir::cast<fir::SliceOp>(slice.getDefiningOp());
    mlir::Operation::operand_range triples = slOp.getTriples();
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Location loc = getLoc();
    for (unsigned i = 0, end = triples.size(); i < end; i += 3) {
      if (!mlir::isa_and_nonnull<fir::UndefOp>(
              triples[i + 1].getDefiningOp())) {
        // (..., lb:ub:step, ...) case:  extent = max((ub-lb+step)/step, 0)
        // See Fortran 2018 9.5.3.3.2 section for more details.
        mlir::Value res = builder.genExtentFromTriplet(
            loc, triples[i], triples[i + 1], triples[i + 2], idxTy);
        slicedShape.emplace_back(res);
      } else {
        // do nothing. `..., i, ...` case, so dimension is dropped.
      }
    }
    return slicedShape;
  }

  /// Get the shape from an ArrayOperand. The shape of the array is adjusted if
  /// the array was sliced.
  llvm::SmallVector<mlir::Value> getShape(ArrayOperand array) {
    if (array.slice)
      return computeSliceShape(array.slice);
    if (array.memref.getType().isa<fir::BaseBoxType>())
      return fir::factory::readExtents(builder, getLoc(),
                                       fir::BoxValue{array.memref});
    return fir::factory::getExtents(array.shape);
  }

  /// Get the shape from an ArrayLoad.
  llvm::SmallVector<mlir::Value> getShape(fir::ArrayLoadOp arrayLoad) {
    return getShape(ArrayOperand{arrayLoad.getMemref(), arrayLoad.getShape(),
                                 arrayLoad.getSlice()});
  }

  /// Returns the first array operand that may not be absent. If all
  /// array operands may be absent, return the first one.
  const ArrayOperand &getInducingShapeArrayOperand() const {
    assert(!arrayOperands.empty());
    for (const ArrayOperand &op : arrayOperands)
      if (!op.mayBeAbsent)
        return op;
    // If all arrays operand appears in optional position, then none of them
    // is allowed to be absent as per 15.5.2.12 point 3. (6). Just pick the
    // first operands.
    // TODO: There is an opportunity to add a runtime check here that
    // this array is present as required.
    return arrayOperands[0];
  }

  /// Generate the shape of the iteration space over the array expression. The
  /// iteration space may be implicit, explicit, or both. If it is implied it is
  /// based on the destination and operand array loads, or an optional
  /// Fortran::evaluate::Shape from the front end. If the shape is explicit,
  /// this returns any implicit shape component, if it exists.
  llvm::SmallVector<mlir::Value> genIterationShape() {
    // Use the precomputed destination shape.
    if (!destShape.empty())
      return destShape;
    // Otherwise, use the destination's shape.
    if (destination)
      return getShape(destination);
    // Otherwise, use the first ArrayLoad operand shape.
    if (!arrayOperands.empty())
      return getShape(getInducingShapeArrayOperand());
    // Otherwise, in elemental context, try to find the passed object and
    // retrieve the iteration shape from it.
    if (loweredProcRef && loweredProcRef->IsElemental()) {
      const std::optional<Fortran::evaluate::ActualArgument> passArg =
          extractPassedArgFromProcRef(*loweredProcRef, converter);
      if (passArg) {
        ExtValue exv = asScalarRef(*passArg->UnwrapExpr());
        fir::FirOpBuilder *builder = &converter.getFirOpBuilder();
        return fir::factory::getExtents(getLoc(), *builder, exv);
      }
    }
    fir::emitFatalError(getLoc(),
                        "failed to compute the array expression shape");
  }

  bool explicitSpaceIsActive() const {
    return explicitSpace && explicitSpace->isActive();
  }

  bool implicitSpaceHasMasks() const {
    return implicitSpace && !implicitSpace->empty();
  }

  CC genMaskAccess(mlir::Value tmp, mlir::Value shape) {
    mlir::Location loc = getLoc();
    return [=, builder = &converter.getFirOpBuilder()](IterSpace iters) {
      mlir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(tmp.getType());
      auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
      mlir::Type eleRefTy = builder->getRefType(eleTy);
      mlir::IntegerType i1Ty = builder->getI1Type();
      // Adjust indices for any shift of the origin of the array.
      llvm::SmallVector<mlir::Value> indices = fir::factory::originateIndices(
          loc, *builder, tmp.getType(), shape, iters.iterVec());
      auto addr =
          builder->create<fir::ArrayCoorOp>(loc, eleRefTy, tmp, shape,
                                            /*slice=*/mlir::Value{}, indices,
                                            /*typeParams=*/std::nullopt);
      auto load = builder->create<fir::LoadOp>(loc, addr);
      return builder->createConvert(loc, i1Ty, load);
    };
  }

  /// Construct the incremental instantiations of the ragged array structure.
  /// Rebind the lazy buffer variable, etc. as we go.
  template <bool withAllocation = false>
  mlir::Value prepareRaggedArrays(Fortran::lower::FrontEndExpr expr) {
    assert(explicitSpaceIsActive());
    mlir::Location loc = getLoc();
    mlir::TupleType raggedTy = fir::factory::getRaggedArrayHeaderType(builder);
    llvm::SmallVector<llvm::SmallVector<fir::DoLoopOp>> loopStack =
        explicitSpace->getLoopStack();
    const std::size_t depth = loopStack.size();
    mlir::IntegerType i64Ty = builder.getIntegerType(64);
    [[maybe_unused]] mlir::Value byteSize =
        builder.createIntegerConstant(loc, i64Ty, 1);
    mlir::Value header = implicitSpace->lookupMaskHeader(expr);
    for (std::remove_const_t<decltype(depth)> i = 0; i < depth; ++i) {
      auto insPt = builder.saveInsertionPoint();
      if (i < depth - 1)
        builder.setInsertionPoint(loopStack[i + 1][0]);

      // Compute and gather the extents.
      llvm::SmallVector<mlir::Value> extents;
      for (auto doLoop : loopStack[i])
        extents.push_back(builder.genExtentFromTriplet(
            loc, doLoop.getLowerBound(), doLoop.getUpperBound(),
            doLoop.getStep(), i64Ty));
      if constexpr (withAllocation) {
        fir::runtime::genRaggedArrayAllocate(
            loc, builder, header, /*asHeader=*/true, byteSize, extents);
      }

      // Compute the dynamic position into the header.
      llvm::SmallVector<mlir::Value> offsets;
      for (auto doLoop : loopStack[i]) {
        auto m = builder.create<mlir::arith::SubIOp>(
            loc, doLoop.getInductionVar(), doLoop.getLowerBound());
        auto n = builder.create<mlir::arith::DivSIOp>(loc, m, doLoop.getStep());
        mlir::Value one = builder.createIntegerConstant(loc, n.getType(), 1);
        offsets.push_back(builder.create<mlir::arith::AddIOp>(loc, n, one));
      }
      mlir::IntegerType i32Ty = builder.getIntegerType(32);
      mlir::Value uno = builder.createIntegerConstant(loc, i32Ty, 1);
      mlir::Type coorTy = builder.getRefType(raggedTy.getType(1));
      auto hdOff = builder.create<fir::CoordinateOp>(loc, coorTy, header, uno);
      auto toTy = fir::SequenceType::get(raggedTy, offsets.size());
      mlir::Type toRefTy = builder.getRefType(toTy);
      auto ldHdr = builder.create<fir::LoadOp>(loc, hdOff);
      mlir::Value hdArr = builder.createConvert(loc, toRefTy, ldHdr);
      auto shapeOp = builder.genShape(loc, extents);
      header = builder.create<fir::ArrayCoorOp>(
          loc, builder.getRefType(raggedTy), hdArr, shapeOp,
          /*slice=*/mlir::Value{}, offsets,
          /*typeparams=*/mlir::ValueRange{});
      auto hdrVar = builder.create<fir::CoordinateOp>(loc, coorTy, header, uno);
      auto inVar = builder.create<fir::LoadOp>(loc, hdrVar);
      mlir::Value two = builder.createIntegerConstant(loc, i32Ty, 2);
      mlir::Type coorTy2 = builder.getRefType(raggedTy.getType(2));
      auto hdrSh = builder.create<fir::CoordinateOp>(loc, coorTy2, header, two);
      auto shapePtr = builder.create<fir::LoadOp>(loc, hdrSh);
      // Replace the binding.
      implicitSpace->rebind(expr, genMaskAccess(inVar, shapePtr));
      if (i < depth - 1)
        builder.restoreInsertionPoint(insPt);
    }
    return header;
  }

  /// Lower mask expressions with implied iteration spaces from the variants of
  /// WHERE syntax. Since it is legal for mask expressions to have side-effects
  /// and modify values that will be used for the lhs, rhs, or both of
  /// subsequent assignments, the mask must be evaluated before the assignment
  /// is processed.
  /// Mask expressions are array expressions too.
  void genMasks() {
    // Lower the mask expressions, if any.
    if (implicitSpaceHasMasks()) {
      mlir::Location loc = getLoc();
      // Mask expressions are array expressions too.
      for (const auto *e : implicitSpace->getExprs())
        if (e && !implicitSpace->isLowered(e)) {
          if (mlir::Value var = implicitSpace->lookupMaskVariable(e)) {
            // Allocate the mask buffer lazily.
            assert(explicitSpaceIsActive());
            mlir::Value header =
                prepareRaggedArrays</*withAllocations=*/true>(e);
            Fortran::lower::createLazyArrayTempValue(converter, *e, header,
                                                     symMap, stmtCtx);
            // Close the explicit loops.
            builder.create<fir::ResultOp>(loc, explicitSpace->getInnerArgs());
            builder.setInsertionPointAfter(explicitSpace->getOuterLoop());
            // Open a new copy of the explicit loop nest.
            explicitSpace->genLoopNest();
            continue;
          }
          fir::ExtendedValue tmp = Fortran::lower::createSomeArrayTempValue(
              converter, *e, symMap, stmtCtx);
          mlir::Value shape = builder.createShape(loc, tmp);
          implicitSpace->bind(e, genMaskAccess(fir::getBase(tmp), shape));
        }

      // Set buffer from the header.
      for (const auto *e : implicitSpace->getExprs()) {
        if (!e)
          continue;
        if (implicitSpace->lookupMaskVariable(e)) {
          // Index into the ragged buffer to retrieve cached results.
          const int rank = e->Rank();
          assert(destShape.empty() ||
                 static_cast<std::size_t>(rank) == destShape.size());
          mlir::Value header = prepareRaggedArrays(e);
          mlir::TupleType raggedTy =
              fir::factory::getRaggedArrayHeaderType(builder);
          mlir::IntegerType i32Ty = builder.getIntegerType(32);
          mlir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
          auto coor1 = builder.create<fir::CoordinateOp>(
              loc, builder.getRefType(raggedTy.getType(1)), header, one);
          auto db = builder.create<fir::LoadOp>(loc, coor1);
          mlir::Type eleTy =
              fir::unwrapSequenceType(fir::unwrapRefType(db.getType()));
          mlir::Type buffTy =
              builder.getRefType(fir::SequenceType::get(eleTy, rank));
          // Address of ragged buffer data.
          mlir::Value buff = builder.createConvert(loc, buffTy, db);

          mlir::Value two = builder.createIntegerConstant(loc, i32Ty, 2);
          auto coor2 = builder.create<fir::CoordinateOp>(
              loc, builder.getRefType(raggedTy.getType(2)), header, two);
          auto shBuff = builder.create<fir::LoadOp>(loc, coor2);
          mlir::IntegerType i64Ty = builder.getIntegerType(64);
          mlir::IndexType idxTy = builder.getIndexType();
          llvm::SmallVector<mlir::Value> extents;
          for (std::remove_const_t<decltype(rank)> i = 0; i < rank; ++i) {
            mlir::Value off = builder.createIntegerConstant(loc, i32Ty, i);
            auto coor = builder.create<fir::CoordinateOp>(
                loc, builder.getRefType(i64Ty), shBuff, off);
            auto ldExt = builder.create<fir::LoadOp>(loc, coor);
            extents.push_back(builder.createConvert(loc, idxTy, ldExt));
          }
          if (destShape.empty())
            destShape = extents;
          // Construct shape of buffer.
          mlir::Value shapeOp = builder.genShape(loc, extents);

          // Replace binding with the local result.
          implicitSpace->rebind(e, genMaskAccess(buff, shapeOp));
        }
      }
    }
  }

  // FIXME: should take multiple inner arguments.
  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint>
  genImplicitLoops(mlir::ValueRange shape, mlir::Value innerArg) {
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
    llvm::SmallVector<mlir::Value> loopUppers;

    // Convert any implied shape to closed interval form. The fir.do_loop will
    // run from 0 to `extent - 1` inclusive.
    for (auto extent : shape)
      loopUppers.push_back(
          builder.create<mlir::arith::SubIOp>(loc, extent, one));

    // Iteration space is created with outermost columns, innermost rows
    llvm::SmallVector<fir::DoLoopOp> loops;

    const std::size_t loopDepth = loopUppers.size();
    llvm::SmallVector<mlir::Value> ivars;

    for (auto i : llvm::enumerate(llvm::reverse(loopUppers))) {
      if (i.index() > 0) {
        assert(!loops.empty());
        builder.setInsertionPointToStart(loops.back().getBody());
      }
      fir::DoLoopOp loop;
      if (innerArg) {
        loop = builder.create<fir::DoLoopOp>(
            loc, zero, i.value(), one, isUnordered(),
            /*finalCount=*/false, mlir::ValueRange{innerArg});
        innerArg = loop.getRegionIterArgs().front();
        if (explicitSpaceIsActive())
          explicitSpace->setInnerArg(0, innerArg);
      } else {
        loop = builder.create<fir::DoLoopOp>(loc, zero, i.value(), one,
                                             isUnordered(),
                                             /*finalCount=*/false);
      }
      ivars.push_back(loop.getInductionVar());
      loops.push_back(loop);
    }

    if (innerArg)
      for (std::remove_const_t<decltype(loopDepth)> i = 0; i + 1 < loopDepth;
           ++i) {
        builder.setInsertionPointToEnd(loops[i].getBody());
        builder.create<fir::ResultOp>(loc, loops[i + 1].getResult(0));
      }

    // Move insertion point to the start of the innermost loop in the nest.
    builder.setInsertionPointToStart(loops.back().getBody());
    // Set `afterLoopNest` to just after the entire loop nest.
    auto currPt = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(loops[0]);
    auto afterLoopNest = builder.saveInsertionPoint();
    builder.restoreInsertionPoint(currPt);

    // Put the implicit loop variables in row to column order to match FIR's
    // Ops. (The loops were constructed from outermost column to innermost
    // row.)
    mlir::Value outerRes = loops[0].getResult(0);
    return {IterationSpace(innerArg, outerRes, llvm::reverse(ivars)),
            afterLoopNest};
  }

  /// Build the iteration space into which the array expression will be lowered.
  /// The resultType is used to create a temporary, if needed.
  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint>
  genIterSpace(mlir::Type resultType) {
    mlir::Location loc = getLoc();
    llvm::SmallVector<mlir::Value> shape = genIterationShape();
    if (!destination) {
      // Allocate storage for the result if it is not already provided.
      destination = createAndLoadSomeArrayTemp(resultType, shape);
    }

    // Generate the lazy mask allocation, if one was given.
    if (ccPrelude)
      (*ccPrelude)(shape);

    // Now handle the implicit loops.
    mlir::Value inner = explicitSpaceIsActive()
                            ? explicitSpace->getInnerArgs().front()
                            : destination.getResult();
    auto [iters, afterLoopNest] = genImplicitLoops(shape, inner);
    mlir::Value innerArg = iters.innerArgument();

    // Generate the mask conditional structure, if there are masks. Unlike the
    // explicit masks, which are interleaved, these mask expression appear in
    // the innermost loop.
    if (implicitSpaceHasMasks()) {
      // Recover the cached condition from the mask buffer.
      auto genCond = [&](Fortran::lower::FrontEndExpr e, IterSpace iters) {
        return implicitSpace->getBoundClosure(e)(iters);
      };

      // Handle the negated conditions in topological order of the WHERE
      // clauses. See 10.2.3.2p4 as to why this control structure is produced.
      for (llvm::SmallVector<Fortran::lower::FrontEndExpr> maskExprs :
           implicitSpace->getMasks()) {
        const std::size_t size = maskExprs.size() - 1;
        auto genFalseBlock = [&](const auto *e, auto &&cond) {
          auto ifOp = builder.create<fir::IfOp>(
              loc, mlir::TypeRange{innerArg.getType()}, fir::getBase(cond),
              /*withElseRegion=*/true);
          builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          builder.create<fir::ResultOp>(loc, innerArg);
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        };
        auto genTrueBlock = [&](const auto *e, auto &&cond) {
          auto ifOp = builder.create<fir::IfOp>(
              loc, mlir::TypeRange{innerArg.getType()}, fir::getBase(cond),
              /*withElseRegion=*/true);
          builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          builder.create<fir::ResultOp>(loc, innerArg);
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        };
        for (std::remove_const_t<decltype(size)> i = 0; i < size; ++i)
          if (const auto *e = maskExprs[i])
            genFalseBlock(e, genCond(e, iters));

        // The last condition is either non-negated or unconditionally negated.
        if (const auto *e = maskExprs[size])
          genTrueBlock(e, genCond(e, iters));
      }
    }

    // We're ready to lower the body (an assignment statement) for this context
    // of loop nests at this point.
    return {iters, afterLoopNest};
  }

  fir::ArrayLoadOp
  createAndLoadSomeArrayTemp(mlir::Type type,
                             llvm::ArrayRef<mlir::Value> shape) {
    mlir::Location loc = getLoc();
    if (fir::isPolymorphicType(type))
      TODO(loc, "polymorphic array temporary");
    if (ccLoadDest)
      return (*ccLoadDest)(shape);
    auto seqTy = type.dyn_cast<fir::SequenceType>();
    assert(seqTy && "must be an array");
    // TODO: Need to thread the LEN parameters here. For character, they may
    // differ from the operands length (e.g concatenation). So the array loads
    // type parameters are not enough.
    if (auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>())
      if (charTy.hasDynamicLen())
        TODO(loc, "character array expression temp with dynamic length");
    if (auto recTy = seqTy.getEleTy().dyn_cast<fir::RecordType>())
      if (recTy.getNumLenParams() > 0)
        TODO(loc, "derived type array expression temp with LEN parameters");
    if (mlir::Type eleTy = fir::unwrapSequenceType(type);
        fir::isRecordWithAllocatableMember(eleTy))
      TODO(loc, "creating an array temp where the element type has "
                "allocatable members");
    mlir::Value temp = !seqTy.hasDynamicExtents()
                           ? builder.create<fir::AllocMemOp>(loc, type)
                           : builder.create<fir::AllocMemOp>(
                                 loc, type, ".array.expr", std::nullopt, shape);
    fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup(
        [bldr, loc, temp]() { bldr->create<fir::FreeMemOp>(loc, temp); });
    mlir::Value shapeOp = genShapeOp(shape);
    return builder.create<fir::ArrayLoadOp>(loc, seqTy, temp, shapeOp,
                                            /*slice=*/mlir::Value{},
                                            std::nullopt);
  }

  static fir::ShapeOp genShapeOp(mlir::Location loc, fir::FirOpBuilder &builder,
                                 llvm::ArrayRef<mlir::Value> shape) {
    mlir::IndexType idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> idxShape;
    for (auto s : shape)
      idxShape.push_back(builder.createConvert(loc, idxTy, s));
    return builder.create<fir::ShapeOp>(loc, idxShape);
  }

  fir::ShapeOp genShapeOp(llvm::ArrayRef<mlir::Value> shape) {
    return genShapeOp(getLoc(), builder, shape);
  }

  //===--------------------------------------------------------------------===//
  // Expression traversal and lowering.
  //===--------------------------------------------------------------------===//

  /// Lower the expression, \p x, in a scalar context.
  template <typename A>
  ExtValue asScalar(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.genval(x);
  }

  /// Lower the expression, \p x, in a scalar context. If this is an explicit
  /// space, the expression may be scalar and refer to an array. We want to
  /// raise the array access to array operations in FIR to analyze potential
  /// conflicts even when the result is a scalar element.
  template <typename A>
  ExtValue asScalarArray(const A &x) {
    return explicitSpaceIsActive() && !isPointerAssignment()
               ? genarr(x)(IterationSpace{})
               : asScalar(x);
  }

  /// Lower the expression in a scalar context to a memory reference.
  template <typename A>
  ExtValue asScalarRef(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.gen(x);
  }

  /// Lower an expression without dereferencing any indirection that may be
  /// a nullptr (because this is an absent optional or unallocated/disassociated
  /// descriptor). The returned expression cannot be addressed directly, it is
  /// meant to inquire about its status before addressing the related entity.
  template <typename A>
  ExtValue asInquired(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}
        .lowerIntrinsicArgumentAsInquired(x);
  }

  /// Some temporaries are allocated on an element-by-element basis during the
  /// array expression evaluation. Collect the cleanups here so the resources
  /// can be freed before the next loop iteration, avoiding memory leaks. etc.
  Fortran::lower::StatementContext &getElementCtx() {
    if (!elementCtx) {
      stmtCtx.pushScope();
      elementCtx = true;
    }
    return stmtCtx;
  }

  /// If there were temporaries created for this element evaluation, finalize
  /// and deallocate the resources now. This should be done just prior the the
  /// fir::ResultOp at the end of the innermost loop.
  void finalizeElementCtx() {
    if (elementCtx) {
      stmtCtx.finalizeAndPop();
      elementCtx = false;
    }
  }

  /// Lower an elemental function array argument. This ensures array
  /// sub-expressions that are not variables and must be passed by address
  /// are lowered by value and placed in memory.
  template <typename A>
  CC genElementalArgument(const A &x) {
    // Ensure the returned element is in memory if this is what was requested.
    if ((semant == ConstituentSemantics::RefOpaque ||
         semant == ConstituentSemantics::DataAddr ||
         semant == ConstituentSemantics::ByValueArg)) {
      if (!Fortran::evaluate::IsVariable(x)) {
        PushSemantics(ConstituentSemantics::DataValue);
        CC cc = genarr(x);
        mlir::Location loc = getLoc();
        if (isParenthesizedVariable(x)) {
          // Parenthesised variables are lowered to a reference to the variable
          // storage. When passing it as an argument, a copy must be passed.
          return [=](IterSpace iters) -> ExtValue {
            return createInMemoryScalarCopy(builder, loc, cc(iters));
          };
        }
        mlir::Type storageType =
            fir::unwrapSequenceType(converter.genType(toEvExpr(x)));
        return [=](IterSpace iters) -> ExtValue {
          return placeScalarValueInMemory(builder, loc, cc(iters), storageType);
        };
      }
    }
    return genarr(x);
  }

  // A reference to a Fortran elemental intrinsic or intrinsic module procedure.
  CC genElementalIntrinsicProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      std::optional<mlir::Type> retTy,
      std::optional<const Fortran::evaluate::SpecificIntrinsic> intrinsic =
          std::nullopt) {

    llvm::SmallVector<CC> operands;
    std::string name =
        intrinsic ? intrinsic->name
                  : procRef.proc().GetSymbol()->GetUltimate().name().ToString();
    const fir::IntrinsicArgumentLoweringRules *argLowering =
        fir::getIntrinsicArgumentLowering(name);
    mlir::Location loc = getLoc();
    if (intrinsic && Fortran::lower::intrinsicRequiresCustomOptionalHandling(
                         procRef, *intrinsic, converter)) {
      using CcPairT = std::pair<CC, std::optional<mlir::Value>>;
      llvm::SmallVector<CcPairT> operands;
      auto prepareOptionalArg = [&](const Fortran::lower::SomeExpr &expr) {
        if (expr.Rank() == 0) {
          ExtValue optionalArg = this->asInquired(expr);
          mlir::Value isPresent =
              genActualIsPresentTest(builder, loc, optionalArg);
          operands.emplace_back(
              [=](IterSpace iters) -> ExtValue {
                return genLoad(builder, loc, optionalArg);
              },
              isPresent);
        } else {
          auto [cc, isPresent, _] = this->genOptionalArrayFetch(expr);
          operands.emplace_back(cc, isPresent);
        }
      };
      auto prepareOtherArg = [&](const Fortran::lower::SomeExpr &expr,
                                 fir::LowerIntrinsicArgAs lowerAs) {
        assert(lowerAs == fir::LowerIntrinsicArgAs::Value &&
               "expect value arguments for elemental intrinsic");
        PushSemantics(ConstituentSemantics::RefTransparent);
        operands.emplace_back(genElementalArgument(expr), std::nullopt);
      };
      Fortran::lower::prepareCustomIntrinsicArgument(
          procRef, *intrinsic, retTy, prepareOptionalArg, prepareOtherArg,
          converter);

      fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
      return [=](IterSpace iters) -> ExtValue {
        auto getArgument = [&](std::size_t i, bool) -> ExtValue {
          return operands[i].first(iters);
        };
        auto isPresent = [&](std::size_t i) -> std::optional<mlir::Value> {
          return operands[i].second;
        };
        return Fortran::lower::lowerCustomIntrinsic(
            *bldr, loc, name, retTy, isPresent, getArgument, operands.size(),
            getElementCtx());
      };
    }
    /// Otherwise, pre-lower arguments and use intrinsic lowering utility.
    for (const auto &arg : llvm::enumerate(procRef.arguments())) {
      const auto *expr =
          Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
      if (!expr) {
        // Absent optional.
        operands.emplace_back([=](IterSpace) { return mlir::Value{}; });
      } else if (!argLowering) {
        // No argument lowering instruction, lower by value.
        PushSemantics(ConstituentSemantics::RefTransparent);
        operands.emplace_back(genElementalArgument(*expr));
      } else {
        // Ad-hoc argument lowering handling.
        fir::ArgLoweringRule argRules =
            fir::lowerIntrinsicArgumentAs(*argLowering, arg.index());
        if (argRules.handleDynamicOptional &&
            Fortran::evaluate::MayBePassedAsAbsentOptional(
                *expr, converter.getFoldingContext())) {
          // Currently, there is not elemental intrinsic that requires lowering
          // a potentially absent argument to something else than a value (apart
          // from character MAX/MIN that are handled elsewhere.)
          if (argRules.lowerAs != fir::LowerIntrinsicArgAs::Value)
            TODO(loc, "non trivial optional elemental intrinsic array "
                      "argument");
          PushSemantics(ConstituentSemantics::RefTransparent);
          operands.emplace_back(genarrForwardOptionalArgumentToCall(*expr));
          continue;
        }
        switch (argRules.lowerAs) {
        case fir::LowerIntrinsicArgAs::Value: {
          PushSemantics(ConstituentSemantics::RefTransparent);
          operands.emplace_back(genElementalArgument(*expr));
        } break;
        case fir::LowerIntrinsicArgAs::Addr: {
          // Note: assume does not have Fortran VALUE attribute semantics.
          PushSemantics(ConstituentSemantics::RefOpaque);
          operands.emplace_back(genElementalArgument(*expr));
        } break;
        case fir::LowerIntrinsicArgAs::Box: {
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genElementalArgument(*expr);
          operands.emplace_back([=](IterSpace iters) {
            return builder.createBox(loc, lambda(iters));
          });
        } break;
        case fir::LowerIntrinsicArgAs::Inquired:
          TODO(loc, "intrinsic function with inquired argument");
          break;
        }
      }
    }

    // Let the intrinsic library lower the intrinsic procedure call
    return [=](IterSpace iters) {
      llvm::SmallVector<ExtValue> args;
      for (const auto &cc : operands)
        args.push_back(cc(iters));
      return Fortran::lower::genIntrinsicCall(builder, loc, name, retTy, args,
                                              getElementCtx());
    };
  }

  /// Lower a procedure reference to a user-defined elemental procedure.
  CC genElementalUserDefinedProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      std::optional<mlir::Type> retTy) {
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    // 10.1.4 p5. Impure elemental procedures must be called in element order.
    if (const Fortran::semantics::Symbol *procSym = procRef.proc().GetSymbol())
      if (!Fortran::semantics::IsPureProcedure(*procSym))
        setUnordered(false);

    Fortran::lower::CallerInterface caller(procRef, converter);
    llvm::SmallVector<CC> operands;
    operands.reserve(caller.getPassedArguments().size());
    mlir::Location loc = getLoc();
    mlir::FunctionType callSiteType = caller.genFunctionType();
    for (const Fortran::lower::CallInterface<
             Fortran::lower::CallerInterface>::PassedEntity &arg :
         caller.getPassedArguments()) {
      // 15.8.3 p1. Elemental procedure with intent(out)/intent(inout)
      // arguments must be called in element order.
      if (arg.mayBeModifiedByCall())
        setUnordered(false);
      const auto *actual = arg.entity;
      mlir::Type argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        auto absent = builder.create<fir::AbsentOp>(loc, argTy);
        operands.emplace_back([=](IterSpace) { return absent; });
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument");

      LLVM_DEBUG(expr->AsFortran(llvm::dbgs()
                                 << "argument: " << arg.firArgument << " = [")
                 << "]\n");
      if (arg.isOptional() && Fortran::evaluate::MayBePassedAsAbsentOptional(
                                  *expr, converter.getFoldingContext()))
        TODO(loc,
             "passing dynamically optional argument to elemental procedures");
      switch (arg.passBy) {
      case PassBy::Value: {
        // True pass-by-value semantics.
        PushSemantics(ConstituentSemantics::RefTransparent);
        operands.emplace_back(genElementalArgument(*expr));
      } break;
      case PassBy::BaseAddressValueAttribute: {
        // VALUE attribute or pass-by-reference to a copy semantics. (byval*)
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::ByValueArg);
          operands.emplace_back(genElementalArgument(*expr));
        } else {
          // Store scalar value in a temp to fulfill VALUE attribute.
          mlir::Value val = fir::getBase(asScalar(*expr));
          mlir::Value temp = builder.createTemporary(
              loc, val.getType(),
              llvm::ArrayRef<mlir::NamedAttribute>{
                  Fortran::lower::getAdaptToByRefAttr(builder)});
          builder.create<fir::StoreOp>(loc, val, temp);
          operands.emplace_back(
              [=](IterSpace iters) -> ExtValue { return temp; });
        }
      } break;
      case PassBy::BaseAddress: {
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::RefOpaque);
          operands.emplace_back(genElementalArgument(*expr));
        } else {
          ExtValue exv = asScalarRef(*expr);
          operands.emplace_back([=](IterSpace iters) { return exv; });
        }
      } break;
      case PassBy::CharBoxValueAttribute: {
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::DataValue);
          auto lambda = genElementalArgument(*expr);
          operands.emplace_back([=](IterSpace iters) {
            return fir::factory::CharacterExprHelper{builder, loc}
                .createTempFrom(lambda(iters));
          });
        } else {
          fir::factory::CharacterExprHelper helper(builder, loc);
          fir::CharBoxValue argVal = helper.createTempFrom(asScalarRef(*expr));
          operands.emplace_back(
              [=](IterSpace iters) -> ExtValue { return argVal; });
        }
      } break;
      case PassBy::BoxChar: {
        PushSemantics(ConstituentSemantics::RefOpaque);
        operands.emplace_back(genElementalArgument(*expr));
      } break;
      case PassBy::AddressAndLength:
        // PassBy::AddressAndLength is only used for character results. Results
        // are not handled here.
        fir::emitFatalError(
            loc, "unexpected PassBy::AddressAndLength in elemental call");
        break;
      case PassBy::CharProcTuple: {
        ExtValue argRef = asScalarRef(*expr);
        mlir::Value tuple = createBoxProcCharTuple(
            converter, argTy, fir::getBase(argRef), fir::getLen(argRef));
        operands.emplace_back(
            [=](IterSpace iters) -> ExtValue { return tuple; });
      } break;
      case PassBy::Box:
      case PassBy::MutableBox:
        // Handle polymorphic passed object.
        if (fir::isPolymorphicType(argTy)) {
          if (isArray(*expr)) {
            ExtValue exv = asScalarRef(*expr);
            mlir::Value sourceBox;
            if (fir::isPolymorphicType(fir::getBase(exv).getType()))
              sourceBox = fir::getBase(exv);
            mlir::Type baseTy =
                fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(exv).getType());
            mlir::Type innerTy = fir::unwrapSequenceType(baseTy);
            operands.emplace_back([=](IterSpace iters) -> ExtValue {
              mlir::Value coord = builder.create<fir::CoordinateOp>(
                  loc, fir::ReferenceType::get(innerTy), fir::getBase(exv),
                  iters.iterVec());
              mlir::Value empty;
              mlir::ValueRange emptyRange;
              return builder.create<fir::EmboxOp>(
                  loc, fir::ClassType::get(innerTy), coord, empty, empty,
                  emptyRange, sourceBox);
            });
          } else {
            ExtValue exv = asScalarRef(*expr);
            if (fir::getBase(exv).getType().isa<fir::BaseBoxType>()) {
              operands.emplace_back(
                  [=](IterSpace iters) -> ExtValue { return exv; });
            } else {
              mlir::Type baseTy =
                  fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(exv).getType());
              operands.emplace_back([=](IterSpace iters) -> ExtValue {
                mlir::Value empty;
                mlir::ValueRange emptyRange;
                return builder.create<fir::EmboxOp>(
                    loc, fir::ClassType::get(baseTy), fir::getBase(exv), empty,
                    empty, emptyRange);
              });
            }
          }
          break;
        }
        // See C15100 and C15101
        fir::emitFatalError(loc, "cannot be POINTER, ALLOCATABLE");
      }
    }

    if (caller.getIfIndirectCallSymbol())
      fir::emitFatalError(loc, "cannot be indirect call");

    // The lambda is mutable so that `caller` copy can be modified inside it.
    return [=,
            caller = std::move(caller)](IterSpace iters) mutable -> ExtValue {
      for (const auto &[cc, argIface] :
           llvm::zip(operands, caller.getPassedArguments())) {
        auto exv = cc(iters);
        auto arg = exv.match(
            [&](const fir::CharBoxValue &cb) -> mlir::Value {
              return fir::factory::CharacterExprHelper{builder, loc}
                  .createEmbox(cb);
            },
            [&](const auto &) { return fir::getBase(exv); });
        caller.placeInput(argIface, arg);
      }
      return Fortran::lower::genCallOpAndResult(
          loc, converter, symMap, getElementCtx(), caller, callSiteType, retTy);
    };
  }

  /// Lower TRANSPOSE call without using runtime TRANSPOSE.
  /// Return continuation for generating the TRANSPOSE result.
  /// The continuation just swaps the iteration space before
  /// invoking continuation for the argument.
  CC genTransposeProcRef(const Fortran::evaluate::ProcedureRef &procRef) {
    assert(procRef.arguments().size() == 1 &&
           "TRANSPOSE must have one argument.");
    const auto *argExpr = procRef.arguments()[0].value().UnwrapExpr();
    assert(argExpr);

    llvm::SmallVector<mlir::Value> savedDestShape = destShape;
    assert((destShape.empty() || destShape.size() == 2) &&
           "TRANSPOSE destination must have rank 2.");

    if (!savedDestShape.empty())
      std::swap(destShape[0], destShape[1]);

    PushSemantics(ConstituentSemantics::RefTransparent);
    llvm::SmallVector<CC> operands{genElementalArgument(*argExpr)};

    if (!savedDestShape.empty()) {
      // If destShape was set before transpose lowering, then
      // restore it. Otherwise, ...
      destShape = savedDestShape;
    } else if (!destShape.empty()) {
      // ... if destShape has been set from the argument lowering,
      // then reverse it.
      assert(destShape.size() == 2 &&
             "TRANSPOSE destination must have rank 2.");
      std::swap(destShape[0], destShape[1]);
    }

    return [=](IterSpace iters) {
      assert(iters.iterVec().size() == 2 &&
             "TRANSPOSE expects 2D iterations space.");
      IterationSpace newIters(iters, {iters.iterValue(1), iters.iterValue(0)});
      return operands.front()(newIters);
    };
  }

  /// Generate a procedure reference. This code is shared for both functions and
  /// subroutines, the difference being reflected by `retTy`.
  CC genProcRef(const Fortran::evaluate::ProcedureRef &procRef,
                std::optional<mlir::Type> retTy) {
    mlir::Location loc = getLoc();
    setLoweredProcRef(&procRef);

    if (isOptimizableTranspose(procRef, converter))
      return genTransposeProcRef(procRef);

    if (procRef.IsElemental()) {
      if (const Fortran::evaluate::SpecificIntrinsic *intrin =
              procRef.proc().GetSpecificIntrinsic()) {
        // All elemental intrinsic functions are pure and cannot modify their
        // arguments. The only elemental subroutine, MVBITS has an Intent(inout)
        // argument. So for this last one, loops must be in element order
        // according to 15.8.3 p1.
        if (!retTy)
          setUnordered(false);

        // Elemental intrinsic call.
        // The intrinsic procedure is called once per element of the array.
        return genElementalIntrinsicProcRef(procRef, retTy, *intrin);
      }
      if (Fortran::lower::isIntrinsicModuleProcRef(procRef))
        return genElementalIntrinsicProcRef(procRef, retTy);
      if (ScalarExprLowering::isStatementFunctionCall(procRef))
        fir::emitFatalError(loc, "statement function cannot be elemental");

      // Elemental call.
      // The procedure is called once per element of the array argument(s).
      return genElementalUserDefinedProcRef(procRef, retTy);
    }

    // Transformational call.
    // The procedure is called once and produces a value of rank > 0.
    if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
            procRef.proc().GetSpecificIntrinsic()) {
      if (explicitSpaceIsActive() && procRef.Rank() == 0) {
        // Elide any implicit loop iters.
        return [=, &procRef](IterSpace) {
          return ScalarExprLowering{loc, converter, symMap, stmtCtx}
              .genIntrinsicRef(procRef, retTy, *intrinsic);
        };
      }
      return genarr(
          ScalarExprLowering{loc, converter, symMap, stmtCtx}.genIntrinsicRef(
              procRef, retTy, *intrinsic));
    }

    const bool isPtrAssn = isPointerAssignment();
    if (explicitSpaceIsActive() && procRef.Rank() == 0) {
      // Elide any implicit loop iters.
      return [=, &procRef](IterSpace) {
        ScalarExprLowering sel(loc, converter, symMap, stmtCtx);
        return isPtrAssn ? sel.genRawProcedureRef(procRef, retTy)
                         : sel.genProcedureRef(procRef, retTy);
      };
    }
    // In the default case, the call can be hoisted out of the loop nest. Apply
    // the iterations to the result, which may be an array value.
    ScalarExprLowering sel(loc, converter, symMap, stmtCtx);
    auto exv = isPtrAssn ? sel.genRawProcedureRef(procRef, retTy)
                         : sel.genProcedureRef(procRef, retTy);
    return genarr(exv);
  }

  CC genarr(const Fortran::evaluate::ProcedureDesignator &) {
    TODO(getLoc(), "procedure designator");
  }
  CC genarr(const Fortran::evaluate::ProcedureRef &x) {
    if (x.hasAlternateReturns())
      fir::emitFatalError(getLoc(),
                          "array procedure reference with alt-return");
    return genProcRef(x, std::nullopt);
  }
  template <typename A>
  CC genScalarAndForwardValue(const A &x) {
    ExtValue result = asScalar(x);
    return [=](IterSpace) { return result; };
  }
  template <typename A, typename = std::enable_if_t<Fortran::common::HasMember<
                            A, Fortran::evaluate::TypelessExpression>>>
  CC genarr(const A &x) {
    return genScalarAndForwardValue(x);
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Expr<A> &x) {
    LLVM_DEBUG(Fortran::lower::DumpEvaluateExpr::dump(llvm::dbgs(), x));
    if (isArray(x) || (explicitSpaceIsActive() && isLeftHandSide()) ||
        isElementalProcWithArrayArgs(x))
      return std::visit([&](const auto &e) { return genarr(e); }, x.u);
    if (explicitSpaceIsActive()) {
      assert(!isArray(x) && !isLeftHandSide());
      auto cc = std::visit([&](const auto &e) { return genarr(e); }, x.u);
      auto result = cc(IterationSpace{});
      return [=](IterSpace) { return result; };
    }
    return genScalarAndForwardValue(x);
  }

  // Converting a value of memory bound type requires creating a temp and
  // copying the value.
  static ExtValue convertAdjustedType(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type toType,
                                      const ExtValue &exv) {
    return exv.match(
        [&](const fir::CharBoxValue &cb) -> ExtValue {
          mlir::Value len = cb.getLen();
          auto mem =
              builder.create<fir::AllocaOp>(loc, toType, mlir::ValueRange{len});
          fir::CharBoxValue result(mem, len);
          fir::factory::CharacterExprHelper{builder, loc}.createAssign(
              ExtValue{result}, exv);
          return result;
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "convert on adjusted extended value");
        });
  }
  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  CC genarr(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                             TC2> &x) {
    mlir::Location loc = getLoc();
    auto lambda = genarr(x.left());
    mlir::Type ty = converter.genType(TC1, KIND);
    return [=](IterSpace iters) -> ExtValue {
      auto exv = lambda(iters);
      mlir::Value val = fir::getBase(exv);
      auto valTy = val.getType();
      if (elementTypeWasAdjusted(valTy) &&
          !(fir::isa_ref_type(valTy) && fir::isa_integer(ty)))
        return convertAdjustedType(builder, loc, ty, exv);
      return builder.createConvert(loc, ty, val);
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    mlir::Location loc = getLoc();
    auto lambda = genarr(x.left());
    bool isImagPart = x.isImaginaryPart;
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value lhs = fir::getBase(lambda(iters));
      return fir::factory::Complex{builder, loc}.extractComplexPart(lhs,
                                                                    isImagPart);
    };
  }

  template <typename T>
  CC genarr(const Fortran::evaluate::Parentheses<T> &x) {
    mlir::Location loc = getLoc();
    if (isReferentiallyOpaque()) {
      // Context is a call argument in, for example, an elemental procedure
      // call. TODO: all array arguments should use array_load, array_access,
      // array_amend, and INTENT(OUT), INTENT(INOUT) arguments should have
      // array_merge_store ops.
      TODO(loc, "parentheses on argument in elemental call");
    }
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = f(iters);
      mlir::Value base = fir::getBase(val);
      auto newBase =
          builder.create<fir::NoReassocOp>(loc, base.getType(), base);
      return fir::substBase(val, newBase);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    mlir::Location loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value val = fir::getBase(f(iters));
      mlir::Type ty =
          converter.genType(Fortran::common::TypeCategory::Integer, KIND);
      mlir::Value zero = builder.createIntegerConstant(loc, ty, 0);
      return builder.create<mlir::arith::SubIOp>(loc, zero, val);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    mlir::Location loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<mlir::arith::NegFOp>(loc, fir::getBase(f(iters)));
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    mlir::Location loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<fir::NegcOp>(loc, fir::getBase(f(iters)));
    };
  }

  //===--------------------------------------------------------------------===//
  // Binary elemental ops
  //===--------------------------------------------------------------------===//

  template <typename OP, typename A>
  CC createBinaryOp(const A &evEx) {
    mlir::Location loc = getLoc();
    auto lambda = genarr(evEx.left());
    auto rf = genarr(evEx.right());
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value left = fir::getBase(lambda(iters));
      mlir::Value right = fir::getBase(rf(iters));
      return builder.create<OP>(loc, left, right);
    };
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  CC genarr(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type<       \
                Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) {       \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::arith::AddIOp)
  GENBIN(Add, Real, mlir::arith::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::arith::SubIOp)
  GENBIN(Subtract, Real, mlir::arith::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::arith::MulIOp)
  GENBIN(Multiply, Real, mlir::arith::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::arith::DivSIOp)
  GENBIN(Divide, Real, mlir::arith::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &x) {
    mlir::Location loc = getLoc();
    mlir::Type ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value lhs = fir::getBase(lf(iters));
      mlir::Value rhs = fir::getBase(rf(iters));
      return fir::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    mlir::Location loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    switch (x.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return [=](IterSpace iters) -> ExtValue {
        mlir::Value lhs = fir::getBase(lf(iters));
        mlir::Value rhs = fir::getBase(rf(iters));
        return fir::genMax(builder, loc, llvm::ArrayRef<mlir::Value>{lhs, rhs});
      };
    case Fortran::evaluate::Ordering::Less:
      return [=](IterSpace iters) -> ExtValue {
        mlir::Value lhs = fir::getBase(lf(iters));
        mlir::Value rhs = fir::getBase(rf(iters));
        return fir::genMin(builder, loc, llvm::ArrayRef<mlir::Value>{lhs, rhs});
      };
    case Fortran::evaluate::Ordering::Equal:
      llvm_unreachable("Equal is not a valid ordering in this context");
    }
    llvm_unreachable("unknown ordering");
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    mlir::Location loc = getLoc();
    auto ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) {
      mlir::Value lhs = fir::getBase(lf(iters));
      mlir::Value rhs = fir::getBase(rf(iters));
      return fir::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    mlir::Location loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value lhs = fir::getBase(lf(iters));
      mlir::Value rhs = fir::getBase(rf(iters));
      return fir::factory::Complex{builder, loc}.createComplex(KIND, lhs, rhs);
    };
  }

  /// Fortran's concatenation operator `//`.
  template <int KIND>
  CC genarr(const Fortran::evaluate::Concat<KIND> &x) {
    mlir::Location loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      const fir::CharBoxValue *lchr = lhs.getCharBox();
      const fir::CharBoxValue *rchr = rhs.getCharBox();
      if (lchr && rchr) {
        return fir::factory::CharacterExprHelper{builder, loc}
            .createConcatenate(*lchr, *rchr);
      }
      TODO(loc, "concat on unexpected extended values");
      return mlir::Value{};
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::SetLength<KIND> &x) {
    auto lf = genarr(x.left());
    mlir::Value rhs = fir::getBase(asScalar(x.right()));
    fir::CharBoxValue temp =
        fir::factory::CharacterExprHelper(builder, getLoc())
            .createCharacterTemp(
                fir::CharacterType::getUnknownLen(builder.getContext(), KIND),
                rhs);
    return [=](IterSpace iters) -> ExtValue {
      fir::factory::CharacterExprHelper(builder, getLoc())
          .createAssign(temp, lf(iters));
      return temp;
    };
  }

  template <typename T>
  CC genarr(const Fortran::evaluate::Constant<T> &x) {
    if (x.Rank() == 0)
      return genScalarAndForwardValue(x);
    return genarr(Fortran::lower::convertConstant(
        converter, getLoc(), x,
        /*outlineBigConstantsInReadOnlyMemory=*/true));
  }

  //===--------------------------------------------------------------------===//
  // A vector subscript expression may be wrapped with a cast to INTEGER*8.
  // Get rid of it here so the vector can be loaded. Add it back when
  // generating the elemental evaluation (inside the loop nest).

  static Fortran::lower::SomeExpr
  ignoreEvConvert(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, 8>> &x) {
    return std::visit([&](const auto &v) { return ignoreEvConvert(v); }, x.u);
  }
  template <Fortran::common::TypeCategory FROM>
  static Fortran::lower::SomeExpr ignoreEvConvert(
      const Fortran::evaluate::Convert<
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>,
          FROM> &x) {
    return toEvExpr(x.left());
  }
  template <typename A>
  static Fortran::lower::SomeExpr ignoreEvConvert(const A &x) {
    return toEvExpr(x);
  }

  //===--------------------------------------------------------------------===//
  // Get the `Se::Symbol*` for the subscript expression, `x`. This symbol can
  // be used to determine the lbound, ubound of the vector.

  template <typename A>
  static const Fortran::semantics::Symbol *
  extractSubscriptSymbol(const Fortran::evaluate::Expr<A> &x) {
    return std::visit([&](const auto &v) { return extractSubscriptSymbol(v); },
                      x.u);
  }
  template <typename A>
  static const Fortran::semantics::Symbol *
  extractSubscriptSymbol(const Fortran::evaluate::Designator<A> &x) {
    return Fortran::evaluate::UnwrapWholeSymbolDataRef(x);
  }
  template <typename A>
  static const Fortran::semantics::Symbol *extractSubscriptSymbol(const A &x) {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//

  /// Get the declared lower bound value of the array `x` in dimension `dim`.
  /// The argument `one` must be an ssa-value for the constant 1.
  mlir::Value getLBound(const ExtValue &x, unsigned dim, mlir::Value one) {
    return fir::factory::readLowerBound(builder, getLoc(), x, dim, one);
  }

  /// Get the declared upper bound value of the array `x` in dimension `dim`.
  /// The argument `one` must be an ssa-value for the constant 1.
  mlir::Value getUBound(const ExtValue &x, unsigned dim, mlir::Value one) {
    mlir::Location loc = getLoc();
    mlir::Value lb = getLBound(x, dim, one);
    mlir::Value extent = fir::factory::readExtent(builder, loc, x, dim);
    auto add = builder.create<mlir::arith::AddIOp>(loc, lb, extent);
    return builder.create<mlir::arith::SubIOp>(loc, add, one);
  }

  /// Return the extent of the boxed array `x` in dimesion `dim`.
  mlir::Value getExtent(const ExtValue &x, unsigned dim) {
    return fir::factory::readExtent(builder, getLoc(), x, dim);
  }

  template <typename A>
  ExtValue genArrayBase(const A &base) {
    ScalarExprLowering sel{getLoc(), converter, symMap, stmtCtx};
    return base.IsSymbol() ? sel.gen(getFirstSym(base))
                           : sel.gen(base.GetComponent());
  }

  template <typename A>
  bool hasEvArrayRef(const A &x) {
    struct HasEvArrayRefHelper
        : public Fortran::evaluate::AnyTraverse<HasEvArrayRefHelper> {
      HasEvArrayRefHelper()
          : Fortran::evaluate::AnyTraverse<HasEvArrayRefHelper>(*this) {}
      using Fortran::evaluate::AnyTraverse<HasEvArrayRefHelper>::operator();
      bool operator()(const Fortran::evaluate::ArrayRef &) const {
        return true;
      }
    } helper;
    return helper(x);
  }

  CC genVectorSubscriptArrayFetch(const Fortran::lower::SomeExpr &expr,
                                  std::size_t dim) {
    PushSemantics(ConstituentSemantics::RefTransparent);
    auto saved = Fortran::common::ScopedSet(explicitSpace, nullptr);
    llvm::SmallVector<mlir::Value> savedDestShape = destShape;
    destShape.clear();
    auto result = genarr(expr);
    if (destShape.empty())
      TODO(getLoc(), "expected vector to have an extent");
    assert(destShape.size() == 1 && "vector has rank > 1");
    if (destShape[0] != savedDestShape[dim]) {
      // Not the same, so choose the smaller value.
      mlir::Location loc = getLoc();
      auto cmp = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, destShape[0],
          savedDestShape[dim]);
      auto sel = builder.create<mlir::arith::SelectOp>(
          loc, cmp, savedDestShape[dim], destShape[0]);
      savedDestShape[dim] = sel;
      destShape = savedDestShape;
    }
    return result;
  }

  /// Generate an access by vector subscript using the index in the iteration
  /// vector at `dim`.
  mlir::Value genAccessByVector(mlir::Location loc, CC genArrFetch,
                                IterSpace iters, std::size_t dim) {
    IterationSpace vecIters(iters,
                            llvm::ArrayRef<mlir::Value>{iters.iterValue(dim)});
    fir::ExtendedValue fetch = genArrFetch(vecIters);
    mlir::IndexType idxTy = builder.getIndexType();
    return builder.createConvert(loc, idxTy, fir::getBase(fetch));
  }

  /// When we have an array reference, the expressions specified in each
  /// dimension may be slice operations (e.g. `i:j:k`), vectors, or simple
  /// (loop-invarianet) scalar expressions. This returns the base entity, the
  /// resulting type, and a continuation to adjust the default iteration space.
  void genSliceIndices(ComponentPath &cmptData, const ExtValue &arrayExv,
                       const Fortran::evaluate::ArrayRef &x, bool atBase) {
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    llvm::SmallVector<mlir::Value> &trips = cmptData.trips;
    LLVM_DEBUG(llvm::dbgs() << "array: " << arrayExv << '\n');
    auto &pc = cmptData.pc;
    const bool useTripsForSlice = !explicitSpaceIsActive();
    const bool createDestShape = destShape.empty();
    bool useSlice = false;
    std::size_t shapeIndex = 0;
    for (auto sub : llvm::enumerate(x.subscript())) {
      const std::size_t subsIndex = sub.index();
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::Triplet &t) {
                mlir::Value lowerBound;
                if (auto optLo = t.lower())
                  lowerBound = fir::getBase(asScalarArray(*optLo));
                else
                  lowerBound = getLBound(arrayExv, subsIndex, one);
                lowerBound = builder.createConvert(loc, idxTy, lowerBound);
                mlir::Value stride = fir::getBase(asScalarArray(t.stride()));
                stride = builder.createConvert(loc, idxTy, stride);
                if (useTripsForSlice || createDestShape) {
                  // Generate a slice operation for the triplet. The first and
                  // second position of the triplet may be omitted, and the
                  // declared lbound and/or ubound expression values,
                  // respectively, should be used instead.
                  trips.push_back(lowerBound);
                  mlir::Value upperBound;
                  if (auto optUp = t.upper())
                    upperBound = fir::getBase(asScalarArray(*optUp));
                  else
                    upperBound = getUBound(arrayExv, subsIndex, one);
                  upperBound = builder.createConvert(loc, idxTy, upperBound);
                  trips.push_back(upperBound);
                  trips.push_back(stride);
                  if (createDestShape) {
                    auto extent = builder.genExtentFromTriplet(
                        loc, lowerBound, upperBound, stride, idxTy);
                    destShape.push_back(extent);
                  }
                  useSlice = true;
                }
                if (!useTripsForSlice) {
                  auto currentPC = pc;
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    mlir::Value impliedIter = newIters.iterValue(subsIndex);
                    // FIXME: must use the lower bound of this component.
                    auto arrLowerBound =
                        atBase ? getLBound(arrayExv, subsIndex, one) : one;
                    auto initial = builder.create<mlir::arith::SubIOp>(
                        loc, lowerBound, arrLowerBound);
                    auto prod = builder.create<mlir::arith::MulIOp>(
                        loc, impliedIter, stride);
                    auto result =
                        builder.create<mlir::arith::AddIOp>(loc, initial, prod);
                    newIters.setIndexValue(subsIndex, result);
                    return newIters;
                  };
                }
                shapeIndex++;
              },
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &ie) {
                const auto &e = ie.value(); // dereference
                if (isArray(e)) {
                  // This is a vector subscript. Use the index values as read
                  // from a vector to determine the temporary array value.
                  // Note: 9.5.3.3.3(3) specifies undefined behavior for
                  // multiple updates to any specific array element through a
                  // vector subscript with replicated values.
                  assert(!isBoxValue() &&
                         "fir.box cannot be created with vector subscripts");
                  // TODO: Avoid creating a new evaluate::Expr here
                  auto arrExpr = ignoreEvConvert(e);
                  if (createDestShape) {
                    destShape.push_back(fir::factory::getExtentAtDimension(
                        loc, builder, arrayExv, subsIndex));
                  }
                  auto genArrFetch =
                      genVectorSubscriptArrayFetch(arrExpr, shapeIndex);
                  auto currentPC = pc;
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    auto val = genAccessByVector(loc, genArrFetch, newIters,
                                                 subsIndex);
                    // Value read from vector subscript array and normalized
                    // using the base array's lower bound value.
                    mlir::Value lb = fir::factory::readLowerBound(
                        builder, loc, arrayExv, subsIndex, one);
                    auto origin = builder.create<mlir::arith::SubIOp>(
                        loc, idxTy, val, lb);
                    newIters.setIndexValue(subsIndex, origin);
                    return newIters;
                  };
                  if (useTripsForSlice) {
                    LLVM_ATTRIBUTE_UNUSED auto vectorSubscriptShape =
                        getShape(arrayOperands.back());
                    auto undef = builder.create<fir::UndefOp>(loc, idxTy);
                    trips.push_back(undef);
                    trips.push_back(undef);
                    trips.push_back(undef);
                  }
                  shapeIndex++;
                } else {
                  // This is a regular scalar subscript.
                  if (useTripsForSlice) {
                    // A regular scalar index, which does not yield an array
                    // section. Use a degenerate slice operation
                    // `(e:undef:undef)` in this dimension as a placeholder.
                    // This does not necessarily change the rank of the original
                    // array, so the iteration space must also be extended to
                    // include this expression in this dimension to adjust to
                    // the array's declared rank.
                    mlir::Value v = fir::getBase(asScalarArray(e));
                    trips.push_back(v);
                    auto undef = builder.create<fir::UndefOp>(loc, idxTy);
                    trips.push_back(undef);
                    trips.push_back(undef);
                    auto currentPC = pc;
                    // Cast `e` to index type.
                    mlir::Value iv = builder.createConvert(loc, idxTy, v);
                    // Normalize `e` by subtracting the declared lbound.
                    mlir::Value lb = fir::factory::readLowerBound(
                        builder, loc, arrayExv, subsIndex, one);
                    mlir::Value ivAdj =
                        builder.create<mlir::arith::SubIOp>(loc, idxTy, iv, lb);
                    // Add lbound adjusted value of `e` to the iteration vector
                    // (except when creating a box because the iteration vector
                    // is empty).
                    if (!isBoxValue())
                      pc = [=](IterSpace iters) {
                        IterationSpace newIters = currentPC(iters);
                        newIters.insertIndexValue(subsIndex, ivAdj);
                        return newIters;
                      };
                  } else {
                    auto currentPC = pc;
                    mlir::Value newValue = fir::getBase(asScalarArray(e));
                    mlir::Value result =
                        builder.createConvert(loc, idxTy, newValue);
                    mlir::Value lb = fir::factory::readLowerBound(
                        builder, loc, arrayExv, subsIndex, one);
                    result = builder.create<mlir::arith::SubIOp>(loc, idxTy,
                                                                 result, lb);
                    pc = [=](IterSpace iters) {
                      IterationSpace newIters = currentPC(iters);
                      newIters.insertIndexValue(subsIndex, result);
                      return newIters;
                    };
                  }
                }
              }},
          sub.value().u);
    }
    if (!useSlice)
      trips.clear();
  }

  static mlir::Type unwrapBoxEleTy(mlir::Type ty) {
    if (auto boxTy = ty.dyn_cast<fir::BaseBoxType>())
      return fir::unwrapRefType(boxTy.getEleTy());
    return ty;
  }

  llvm::SmallVector<mlir::Value> getShape(mlir::Type ty) {
    llvm::SmallVector<mlir::Value> result;
    ty = unwrapBoxEleTy(ty);
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    for (auto extent : ty.cast<fir::SequenceType>().getShape()) {
      auto v = extent == fir::SequenceType::getUnknownExtent()
                   ? builder.create<fir::UndefOp>(loc, idxTy).getResult()
                   : builder.createIntegerConstant(loc, idxTy, extent);
      result.push_back(v);
    }
    return result;
  }

  CC genarr(const Fortran::semantics::SymbolRef &sym,
            ComponentPath &components) {
    return genarr(sym.get(), components);
  }

  ExtValue abstractArrayExtValue(mlir::Value val, mlir::Value len = {}) {
    return convertToArrayBoxValue(getLoc(), builder, val, len);
  }

  CC genarr(const ExtValue &extMemref) {
    ComponentPath dummy(/*isImplicit=*/true);
    return genarr(extMemref, dummy);
  }

  // If the slice values are given then use them. Otherwise, generate triples
  // that cover the entire shape specified by \p shapeVal.
  inline llvm::SmallVector<mlir::Value>
  padSlice(llvm::ArrayRef<mlir::Value> triples, mlir::Value shapeVal) {
    llvm::SmallVector<mlir::Value> result;
    mlir::Location loc = getLoc();
    if (triples.size()) {
      result.assign(triples.begin(), triples.end());
    } else {
      auto one = builder.createIntegerConstant(loc, builder.getIndexType(), 1);
      if (!shapeVal) {
        TODO(loc, "shape must be recovered from box");
      } else if (auto shapeOp = mlir::dyn_cast_or_null<fir::ShapeOp>(
                     shapeVal.getDefiningOp())) {
        for (auto ext : shapeOp.getExtents()) {
          result.push_back(one);
          result.push_back(ext);
          result.push_back(one);
        }
      } else if (auto shapeShift = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(
                     shapeVal.getDefiningOp())) {
        for (auto [lb, ext] :
             llvm::zip(shapeShift.getOrigins(), shapeShift.getExtents())) {
          result.push_back(lb);
          result.push_back(ext);
          result.push_back(one);
        }
      } else {
        TODO(loc, "shape must be recovered from box");
      }
    }
    return result;
  }

  /// Base case of generating an array reference,
  CC genarr(const ExtValue &extMemref, ComponentPath &components) {
    mlir::Location loc = getLoc();
    mlir::Value memref = fir::getBase(extMemref);
    mlir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(memref.getType());
    assert(arrTy.isa<fir::SequenceType>() && "memory ref must be an array");
    mlir::Value shape = builder.createShape(loc, extMemref);
    mlir::Value slice;
    if (components.isSlice()) {
      if (isBoxValue() && components.substring) {
        // Append the substring operator to emboxing Op as it will become an
        // interior adjustment (add offset, adjust LEN) to the CHARACTER value
        // being referenced in the descriptor.
        llvm::SmallVector<mlir::Value> substringBounds;
        populateBounds(substringBounds, components.substring);
        // Convert to (offset, size)
        mlir::Type iTy = substringBounds[0].getType();
        if (substringBounds.size() != 2) {
          fir::CharacterType charTy =
              fir::factory::CharacterExprHelper::getCharType(arrTy);
          if (charTy.hasConstantLen()) {
            mlir::IndexType idxTy = builder.getIndexType();
            fir::CharacterType::LenType charLen = charTy.getLen();
            mlir::Value lenValue =
                builder.createIntegerConstant(loc, idxTy, charLen);
            substringBounds.push_back(lenValue);
          } else {
            llvm::SmallVector<mlir::Value> typeparams =
                fir::getTypeParams(extMemref);
            substringBounds.push_back(typeparams.back());
          }
        }
        // Convert the lower bound to 0-based substring.
        mlir::Value one =
            builder.createIntegerConstant(loc, substringBounds[0].getType(), 1);
        substringBounds[0] =
            builder.create<mlir::arith::SubIOp>(loc, substringBounds[0], one);
        // Convert the upper bound to a length.
        mlir::Value cast = builder.createConvert(loc, iTy, substringBounds[1]);
        mlir::Value zero = builder.createIntegerConstant(loc, iTy, 0);
        auto size =
            builder.create<mlir::arith::SubIOp>(loc, cast, substringBounds[0]);
        auto cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sgt, size, zero);
        // size = MAX(upper - (lower - 1), 0)
        substringBounds[1] =
            builder.create<mlir::arith::SelectOp>(loc, cmp, size, zero);
        slice = builder.create<fir::SliceOp>(
            loc, padSlice(components.trips, shape), components.suffixComponents,
            substringBounds);
      } else {
        slice = builder.createSlice(loc, extMemref, components.trips,
                                    components.suffixComponents);
      }
      if (components.hasComponents()) {
        auto seqTy = arrTy.cast<fir::SequenceType>();
        mlir::Type eleTy =
            fir::applyPathToType(seqTy.getEleTy(), components.suffixComponents);
        if (!eleTy)
          fir::emitFatalError(loc, "slicing path is ill-formed");
        if (auto realTy = eleTy.dyn_cast<fir::RealType>())
          eleTy = Fortran::lower::convertReal(realTy.getContext(),
                                              realTy.getFKind());

        // create the type of the projected array.
        arrTy = fir::SequenceType::get(seqTy.getShape(), eleTy);
        LLVM_DEBUG(llvm::dbgs()
                   << "type of array projection from component slicing: "
                   << eleTy << ", " << arrTy << '\n');
      }
    }
    arrayOperands.push_back(ArrayOperand{memref, shape, slice});
    if (destShape.empty())
      destShape = getShape(arrayOperands.back());
    if (isBoxValue()) {
      // Semantics are a reference to a boxed array.
      // This case just requires that an embox operation be created to box the
      // value. The value of the box is forwarded in the continuation.
      mlir::Type reduceTy = reduceRank(arrTy, slice);
      mlir::Type boxTy = fir::BoxType::get(reduceTy);
      if (memref.getType().isa<fir::ClassType>() && !components.hasComponents())
        boxTy = fir::ClassType::get(reduceTy);
      if (components.substring) {
        // Adjust char length to substring size.
        fir::CharacterType charTy =
            fir::factory::CharacterExprHelper::getCharType(reduceTy);
        auto seqTy = reduceTy.cast<fir::SequenceType>();
        // TODO: Use a constant for fir.char LEN if we can compute it.
        boxTy = fir::BoxType::get(
            fir::SequenceType::get(fir::CharacterType::getUnknownLen(
                                       builder.getContext(), charTy.getFKind()),
                                   seqTy.getDimension()));
      }
      llvm::SmallVector<mlir::Value> lbounds;
      llvm::SmallVector<mlir::Value> nonDeferredLenParams;
      if (!slice) {
        lbounds =
            fir::factory::getNonDefaultLowerBounds(builder, loc, extMemref);
        nonDeferredLenParams = fir::factory::getNonDeferredLenParams(extMemref);
      }
      mlir::Value embox =
          memref.getType().isa<fir::BaseBoxType>()
              ? builder.create<fir::ReboxOp>(loc, boxTy, memref, shape, slice)
                    .getResult()
              : builder
                    .create<fir::EmboxOp>(loc, boxTy, memref, shape, slice,
                                          fir::getTypeParams(extMemref))
                    .getResult();
      return [=](IterSpace) -> ExtValue {
        return fir::BoxValue(embox, lbounds, nonDeferredLenParams);
      };
    }
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    if (isReferentiallyOpaque()) {
      // Semantics are an opaque reference to an array.
      // This case forwards a continuation that will generate the address
      // arithmetic to the array element. This does not have copy-in/copy-out
      // semantics. No attempt to copy the array value will be made during the
      // interpretation of the Fortran statement.
      mlir::Type refEleTy = builder.getRefType(eleTy);
      return [=](IterSpace iters) -> ExtValue {
        // ArrayCoorOp does not expect zero based indices.
        llvm::SmallVector<mlir::Value> indices = fir::factory::originateIndices(
            loc, builder, memref.getType(), shape, iters.iterVec());
        mlir::Value coor = builder.create<fir::ArrayCoorOp>(
            loc, refEleTy, memref, shape, slice, indices,
            fir::getTypeParams(extMemref));
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          llvm::SmallVector<mlir::Value> substringBounds;
          populateBounds(substringBounds, components.substring);
          if (!substringBounds.empty()) {
            mlir::Value dstLen = fir::factory::genLenOfCharacter(
                builder, loc, arrTy.cast<fir::SequenceType>(), memref,
                fir::getTypeParams(extMemref), iters.iterVec(),
                substringBounds);
            fir::CharBoxValue dstChar(coor, dstLen);
            return fir::factory::CharacterExprHelper{builder, loc}
                .createSubstring(dstChar, substringBounds);
          }
        }
        return fir::factory::arraySectionElementToExtendedValue(
            builder, loc, extMemref, coor, slice);
      };
    }
    auto arrLoad = builder.create<fir::ArrayLoadOp>(
        loc, arrTy, memref, shape, slice, fir::getTypeParams(extMemref));
    mlir::Value arrLd = arrLoad.getResult();
    if (isProjectedCopyInCopyOut()) {
      // Semantics are projected copy-in copy-out.
      // The backing store of the destination of an array expression may be
      // partially modified. These updates are recorded in FIR by forwarding a
      // continuation that generates an `array_update` Op. The destination is
      // always loaded at the beginning of the statement and merged at the
      // end.
      destination = arrLoad;
      auto lambda = ccStoreToDest
                        ? *ccStoreToDest
                        : defaultStoreToDestination(components.substring);
      return [=](IterSpace iters) -> ExtValue { return lambda(iters); };
    }
    if (isCustomCopyInCopyOut()) {
      // Create an array_modify to get the LHS element address and indicate
      // the assignment, the actual assignment must be implemented in
      // ccStoreToDest.
      destination = arrLoad;
      return [=](IterSpace iters) -> ExtValue {
        mlir::Value innerArg = iters.innerArgument();
        mlir::Type resTy = innerArg.getType();
        mlir::Type eleTy = fir::applyPathToType(resTy, iters.iterVec());
        mlir::Type refEleTy =
            fir::isa_ref_type(eleTy) ? eleTy : builder.getRefType(eleTy);
        auto arrModify = builder.create<fir::ArrayModifyOp>(
            loc, mlir::TypeRange{refEleTy, resTy}, innerArg, iters.iterVec(),
            destination.getTypeparams());
        return abstractArrayExtValue(arrModify.getResult(1));
      };
    }
    if (isCopyInCopyOut()) {
      // Semantics are copy-in copy-out.
      // The continuation simply forwards the result of the `array_load` Op,
      // which is the value of the array as it was when loaded. All data
      // references with rank > 0 in an array expression typically have
      // copy-in copy-out semantics.
      return [=](IterSpace) -> ExtValue { return arrLd; };
    }
    llvm::SmallVector<mlir::Value> arrLdTypeParams =
        fir::factory::getTypeParams(loc, builder, arrLoad);
    if (isValueAttribute()) {
      // Semantics are value attribute.
      // Here the continuation will `array_fetch` a value from an array and
      // then store that value in a temporary. One can thus imitate pass by
      // value even when the call is pass by reference.
      return [=](IterSpace iters) -> ExtValue {
        mlir::Value base;
        mlir::Type eleTy = fir::applyPathToType(arrTy, iters.iterVec());
        if (isAdjustedArrayElementType(eleTy)) {
          mlir::Type eleRefTy = builder.getRefType(eleTy);
          base = builder.create<fir::ArrayAccessOp>(
              loc, eleRefTy, arrLd, iters.iterVec(), arrLdTypeParams);
        } else {
          base = builder.create<fir::ArrayFetchOp>(
              loc, eleTy, arrLd, iters.iterVec(), arrLdTypeParams);
        }
        mlir::Value temp = builder.createTemporary(
            loc, base.getType(),
            llvm::ArrayRef<mlir::NamedAttribute>{
                Fortran::lower::getAdaptToByRefAttr(builder)});
        builder.create<fir::StoreOp>(loc, base, temp);
        return fir::factory::arraySectionElementToExtendedValue(
            builder, loc, extMemref, temp, slice);
      };
    }
    // In the default case, the array reference forwards an `array_fetch` or
    // `array_access` Op in the continuation.
    return [=](IterSpace iters) -> ExtValue {
      mlir::Type eleTy = fir::applyPathToType(arrTy, iters.iterVec());
      if (isAdjustedArrayElementType(eleTy)) {
        mlir::Type eleRefTy = builder.getRefType(eleTy);
        mlir::Value arrayOp = builder.create<fir::ArrayAccessOp>(
            loc, eleRefTy, arrLd, iters.iterVec(), arrLdTypeParams);
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          llvm::SmallVector<mlir::Value> substringBounds;
          populateBounds(substringBounds, components.substring);
          if (!substringBounds.empty()) {
            mlir::Value dstLen = fir::factory::genLenOfCharacter(
                builder, loc, arrLoad, iters.iterVec(), substringBounds);
            fir::CharBoxValue dstChar(arrayOp, dstLen);
            return fir::factory::CharacterExprHelper{builder, loc}
                .createSubstring(dstChar, substringBounds);
          }
        }
        return fir::factory::arraySectionElementToExtendedValue(
            builder, loc, extMemref, arrayOp, slice);
      }
      auto arrFetch = builder.create<fir::ArrayFetchOp>(
          loc, eleTy, arrLd, iters.iterVec(), arrLdTypeParams);
      return fir::factory::arraySectionElementToExtendedValue(
          builder, loc, extMemref, arrFetch, slice);
    };
  }

  std::tuple<CC, mlir::Value, mlir::Type>
  genOptionalArrayFetch(const Fortran::lower::SomeExpr &expr) {
    assert(expr.Rank() > 0 && "expr must be an array");
    mlir::Location loc = getLoc();
    ExtValue optionalArg = asInquired(expr);
    mlir::Value isPresent = genActualIsPresentTest(builder, loc, optionalArg);
    // Generate an array load and access to an array that may be an absent
    // optional or an unallocated optional.
    mlir::Value base = getBase(optionalArg);
    const bool hasOptionalAttr =
        fir::valueHasFirAttribute(base, fir::getOptionalAttrName());
    mlir::Type baseType = fir::unwrapRefType(base.getType());
    const bool isBox = baseType.isa<fir::BoxType>();
    const bool isAllocOrPtr = Fortran::evaluate::IsAllocatableOrPointerObject(
        expr, converter.getFoldingContext());
    mlir::Type arrType = fir::unwrapPassByRefType(baseType);
    mlir::Type eleType = fir::unwrapSequenceType(arrType);
    ExtValue exv = optionalArg;
    if (hasOptionalAttr && isBox && !isAllocOrPtr) {
      // Elemental argument cannot be allocatable or pointers (C15100).
      // Hence, per 15.5.2.12 3 (8) and (9), the provided Allocatable and
      // Pointer optional arrays cannot be absent. The only kind of entities
      // that can get here are optional assumed shape and polymorphic entities.
      exv = absentBoxToUnallocatedBox(builder, loc, exv, isPresent);
    }
    // All the properties can be read from any fir.box but the read values may
    // be undefined and should only be used inside a fir.if (canBeRead) region.
    if (const auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>())
      exv = fir::factory::genMutableBoxRead(builder, loc, *mutableBox);

    mlir::Value memref = fir::getBase(exv);
    mlir::Value shape = builder.createShape(loc, exv);
    mlir::Value noSlice;
    auto arrLoad = builder.create<fir::ArrayLoadOp>(
        loc, arrType, memref, shape, noSlice, fir::getTypeParams(exv));
    mlir::Operation::operand_range arrLdTypeParams = arrLoad.getTypeparams();
    mlir::Value arrLd = arrLoad.getResult();
    // Mark the load to tell later passes it is unsafe to use this array_load
    // shape unconditionally.
    arrLoad->setAttr(fir::getOptionalAttrName(), builder.getUnitAttr());

    // Place the array as optional on the arrayOperands stack so that its
    // shape will only be used as a fallback to induce the implicit loop nest
    // (that is if there is no non optional array arguments).
    arrayOperands.push_back(
        ArrayOperand{memref, shape, noSlice, /*mayBeAbsent=*/true});

    // By value semantics.
    auto cc = [=](IterSpace iters) -> ExtValue {
      auto arrFetch = builder.create<fir::ArrayFetchOp>(
          loc, eleType, arrLd, iters.iterVec(), arrLdTypeParams);
      return fir::factory::arraySectionElementToExtendedValue(
          builder, loc, exv, arrFetch, noSlice);
    };
    return {cc, isPresent, eleType};
  }

  /// Generate a continuation to pass \p expr to an OPTIONAL argument of an
  /// elemental procedure. This is meant to handle the cases where \p expr might
  /// be dynamically absent (i.e. when it is a POINTER, an ALLOCATABLE or an
  /// OPTIONAL variable). If p\ expr is guaranteed to be present genarr() can
  /// directly be called instead.
  CC genarrForwardOptionalArgumentToCall(const Fortran::lower::SomeExpr &expr) {
    mlir::Location loc = getLoc();
    // Only by-value numerical and logical so far.
    if (semant != ConstituentSemantics::RefTransparent)
      TODO(loc, "optional arguments in user defined elemental procedures");

    // Handle scalar argument case (the if-then-else is generated outside of the
    // implicit loop nest).
    if (expr.Rank() == 0) {
      ExtValue optionalArg = asInquired(expr);
      mlir::Value isPresent = genActualIsPresentTest(builder, loc, optionalArg);
      mlir::Value elementValue =
          fir::getBase(genOptionalValue(builder, loc, optionalArg, isPresent));
      return [=](IterSpace iters) -> ExtValue { return elementValue; };
    }

    CC cc;
    mlir::Value isPresent;
    mlir::Type eleType;
    std::tie(cc, isPresent, eleType) = genOptionalArrayFetch(expr);
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value elementValue =
          builder
              .genIfOp(loc, {eleType}, isPresent,
                       /*withElseRegion=*/true)
              .genThen([&]() {
                builder.create<fir::ResultOp>(loc, fir::getBase(cc(iters)));
              })
              .genElse([&]() {
                mlir::Value zero =
                    fir::factory::createZeroValue(builder, loc, eleType);
                builder.create<fir::ResultOp>(loc, zero);
              })
              .getResults()[0];
      return elementValue;
    };
  }

  /// Reduce the rank of a array to be boxed based on the slice's operands.
  static mlir::Type reduceRank(mlir::Type arrTy, mlir::Value slice) {
    if (slice) {
      auto slOp = mlir::dyn_cast<fir::SliceOp>(slice.getDefiningOp());
      assert(slOp && "expected slice op");
      auto seqTy = arrTy.dyn_cast<fir::SequenceType>();
      assert(seqTy && "expected array type");
      mlir::Operation::operand_range triples = slOp.getTriples();
      fir::SequenceType::Shape shape;
      // reduce the rank for each invariant dimension
      for (unsigned i = 1, end = triples.size(); i < end; i += 3) {
        if (auto extent = fir::factory::getExtentFromTriplet(
                triples[i - 1], triples[i], triples[i + 1]))
          shape.push_back(*extent);
        else if (!mlir::isa_and_nonnull<fir::UndefOp>(
                     triples[i].getDefiningOp()))
          shape.push_back(fir::SequenceType::getUnknownExtent());
      }
      return fir::SequenceType::get(shape, seqTy.getEleTy());
    }
    // not sliced, so no change in rank
    return arrTy;
  }

  /// Example: <code>array%RE</code>
  CC genarr(const Fortran::evaluate::ComplexPart &x,
            ComponentPath &components) {
    components.reversePath.push_back(&x);
    return genarr(x.complex(), components);
  }

  template <typename A>
  CC genSlicePath(const A &x, ComponentPath &components) {
    return genarr(x, components);
  }

  CC genarr(const Fortran::evaluate::StaticDataObject::Pointer &,
            ComponentPath &components) {
    TODO(getLoc(), "substring of static object inside FORALL");
  }

  /// Substrings (see 9.4.1)
  CC genarr(const Fortran::evaluate::Substring &x, ComponentPath &components) {
    components.substring = &x;
    return std::visit([&](const auto &v) { return genarr(v, components); },
                      x.parent());
  }

  template <typename T>
  CC genarr(const Fortran::evaluate::FunctionRef<T> &funRef) {
    // Note that it's possible that the function being called returns either an
    // array or a scalar.  In the first case, use the element type of the array.
    return genProcRef(
        funRef, fir::unwrapSequenceType(converter.genType(toEvExpr(funRef))));
  }

  //===--------------------------------------------------------------------===//
  // Array construction
  //===--------------------------------------------------------------------===//

  /// Target agnostic computation of the size of an element in the array.
  /// Returns the size in bytes with type `index` or a null Value if the element
  /// size is not constant.
  mlir::Value computeElementSize(const ExtValue &exv, mlir::Type eleTy,
                                 mlir::Type resTy) {
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value multiplier = builder.createIntegerConstant(loc, idxTy, 1);
    if (fir::hasDynamicSize(eleTy)) {
      if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
        // Array of char with dynamic LEN parameter. Downcast to an array
        // of singleton char, and scale by the len type parameter from
        // `exv`.
        exv.match(
            [&](const fir::CharBoxValue &cb) { multiplier = cb.getLen(); },
            [&](const fir::CharArrayBoxValue &cb) { multiplier = cb.getLen(); },
            [&](const fir::BoxValue &box) {
              multiplier = fir::factory::CharacterExprHelper(builder, loc)
                               .readLengthFromBox(box.getAddr());
            },
            [&](const fir::MutableBoxValue &box) {
              multiplier = fir::factory::CharacterExprHelper(builder, loc)
                               .readLengthFromBox(box.getAddr());
            },
            [&](const auto &) {
              fir::emitFatalError(loc,
                                  "array constructor element has unknown size");
            });
        fir::CharacterType newEleTy = fir::CharacterType::getSingleton(
            eleTy.getContext(), charTy.getFKind());
        if (auto seqTy = resTy.dyn_cast<fir::SequenceType>()) {
          assert(eleTy == seqTy.getEleTy());
          resTy = fir::SequenceType::get(seqTy.getShape(), newEleTy);
        }
        eleTy = newEleTy;
      } else {
        TODO(loc, "dynamic sized type");
      }
    }
    mlir::Type eleRefTy = builder.getRefType(eleTy);
    mlir::Type resRefTy = builder.getRefType(resTy);
    mlir::Value nullPtr = builder.createNullConstant(loc, resRefTy);
    auto offset = builder.create<fir::CoordinateOp>(
        loc, eleRefTy, nullPtr, mlir::ValueRange{multiplier});
    return builder.createConvert(loc, idxTy, offset);
  }

  /// Get the function signature of the LLVM memcpy intrinsic.
  mlir::FunctionType memcpyType() {
    return fir::factory::getLlvmMemcpy(builder).getFunctionType();
  }

  /// Create a call to the LLVM memcpy intrinsic.
  void createCallMemcpy(llvm::ArrayRef<mlir::Value> args) {
    mlir::Location loc = getLoc();
    mlir::func::FuncOp memcpyFunc = fir::factory::getLlvmMemcpy(builder);
    mlir::SymbolRefAttr funcSymAttr =
        builder.getSymbolRefAttr(memcpyFunc.getName());
    mlir::FunctionType funcTy = memcpyFunc.getFunctionType();
    builder.create<fir::CallOp>(loc, funcTy.getResults(), funcSymAttr, args);
  }

  // Construct code to check for a buffer overrun and realloc the buffer when
  // space is depleted. This is done between each item in the ac-value-list.
  mlir::Value growBuffer(mlir::Value mem, mlir::Value needed,
                         mlir::Value bufferSize, mlir::Value buffSize,
                         mlir::Value eleSz) {
    mlir::Location loc = getLoc();
    mlir::func::FuncOp reallocFunc = fir::factory::getRealloc(builder);
    auto cond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sle, bufferSize, needed);
    auto ifOp = builder.create<fir::IfOp>(loc, mem.getType(), cond,
                                          /*withElseRegion=*/true);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    // Not enough space, resize the buffer.
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value two = builder.createIntegerConstant(loc, idxTy, 2);
    auto newSz = builder.create<mlir::arith::MulIOp>(loc, needed, two);
    builder.create<fir::StoreOp>(loc, newSz, buffSize);
    mlir::Value byteSz = builder.create<mlir::arith::MulIOp>(loc, newSz, eleSz);
    mlir::SymbolRefAttr funcSymAttr =
        builder.getSymbolRefAttr(reallocFunc.getName());
    mlir::FunctionType funcTy = reallocFunc.getFunctionType();
    auto newMem = builder.create<fir::CallOp>(
        loc, funcTy.getResults(), funcSymAttr,
        llvm::ArrayRef<mlir::Value>{
            builder.createConvert(loc, funcTy.getInputs()[0], mem),
            builder.createConvert(loc, funcTy.getInputs()[1], byteSz)});
    mlir::Value castNewMem =
        builder.createConvert(loc, mem.getType(), newMem.getResult(0));
    builder.create<fir::ResultOp>(loc, castNewMem);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    // Otherwise, just forward the buffer.
    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    return ifOp.getResult(0);
  }

  /// Copy the next value (or vector of values) into the array being
  /// constructed.
  mlir::Value copyNextArrayCtorSection(const ExtValue &exv, mlir::Value buffPos,
                                       mlir::Value buffSize, mlir::Value mem,
                                       mlir::Value eleSz, mlir::Type eleTy,
                                       mlir::Type eleRefTy, mlir::Type resTy) {
    mlir::Location loc = getLoc();
    auto off = builder.create<fir::LoadOp>(loc, buffPos);
    auto limit = builder.create<fir::LoadOp>(loc, buffSize);
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);

    if (fir::isRecordWithAllocatableMember(eleTy))
      TODO(loc, "deep copy on allocatable members");

    if (!eleSz) {
      // Compute the element size at runtime.
      assert(fir::hasDynamicSize(eleTy));
      if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
        auto charBytes =
            builder.getKindMap().getCharacterBitsize(charTy.getFKind()) / 8;
        mlir::Value bytes =
            builder.createIntegerConstant(loc, idxTy, charBytes);
        mlir::Value length = fir::getLen(exv);
        if (!length)
          fir::emitFatalError(loc, "result is not boxed character");
        eleSz = builder.create<mlir::arith::MulIOp>(loc, bytes, length);
      } else {
        TODO(loc, "PDT size");
        // Will call the PDT's size function with the type parameters.
      }
    }

    // Compute the coordinate using `fir.coordinate_of`, or, if the type has
    // dynamic size, generating the pointer arithmetic.
    auto computeCoordinate = [&](mlir::Value buff, mlir::Value off) {
      mlir::Type refTy = eleRefTy;
      if (fir::hasDynamicSize(eleTy)) {
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          // Scale a simple pointer using dynamic length and offset values.
          auto chTy = fir::CharacterType::getSingleton(charTy.getContext(),
                                                       charTy.getFKind());
          refTy = builder.getRefType(chTy);
          mlir::Type toTy = builder.getRefType(builder.getVarLenSeqTy(chTy));
          buff = builder.createConvert(loc, toTy, buff);
          off = builder.create<mlir::arith::MulIOp>(loc, off, eleSz);
        } else {
          TODO(loc, "PDT offset");
        }
      }
      auto coor = builder.create<fir::CoordinateOp>(loc, refTy, buff,
                                                    mlir::ValueRange{off});
      return builder.createConvert(loc, eleRefTy, coor);
    };

    // Lambda to lower an abstract array box value.
    auto doAbstractArray = [&](const auto &v) {
      // Compute the array size.
      mlir::Value arrSz = one;
      for (auto ext : v.getExtents())
        arrSz = builder.create<mlir::arith::MulIOp>(loc, arrSz, ext);

      // Grow the buffer as needed.
      auto endOff = builder.create<mlir::arith::AddIOp>(loc, off, arrSz);
      mem = growBuffer(mem, endOff, limit, buffSize, eleSz);

      // Copy the elements to the buffer.
      mlir::Value byteSz =
          builder.create<mlir::arith::MulIOp>(loc, arrSz, eleSz);
      auto buff = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
      mlir::Value buffi = computeCoordinate(buff, off);
      llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
          builder, loc, memcpyType(), buffi, v.getAddr(), byteSz,
          /*volatile=*/builder.createBool(loc, false));
      createCallMemcpy(args);

      // Save the incremented buffer position.
      builder.create<fir::StoreOp>(loc, endOff, buffPos);
    };

    // Copy a trivial scalar value into the buffer.
    auto doTrivialScalar = [&](const ExtValue &v, mlir::Value len = {}) {
      // Increment the buffer position.
      auto plusOne = builder.create<mlir::arith::AddIOp>(loc, off, one);

      // Grow the buffer as needed.
      mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

      // Store the element in the buffer.
      mlir::Value buff =
          builder.createConvert(loc, fir::HeapType::get(resTy), mem);
      auto buffi = builder.create<fir::CoordinateOp>(loc, eleRefTy, buff,
                                                     mlir::ValueRange{off});
      fir::factory::genScalarAssignment(
          builder, loc,
          [&]() -> ExtValue {
            if (len)
              return fir::CharBoxValue(buffi, len);
            return buffi;
          }(),
          v);
      builder.create<fir::StoreOp>(loc, plusOne, buffPos);
    };

    // Copy the value.
    exv.match(
        [&](mlir::Value) { doTrivialScalar(exv); },
        [&](const fir::CharBoxValue &v) {
          auto buffer = v.getBuffer();
          if (fir::isa_char(buffer.getType())) {
            doTrivialScalar(exv, eleSz);
          } else {
            // Increment the buffer position.
            auto plusOne = builder.create<mlir::arith::AddIOp>(loc, off, one);

            // Grow the buffer as needed.
            mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

            // Store the element in the buffer.
            mlir::Value buff =
                builder.createConvert(loc, fir::HeapType::get(resTy), mem);
            mlir::Value buffi = computeCoordinate(buff, off);
            llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
                builder, loc, memcpyType(), buffi, v.getAddr(), eleSz,
                /*volatile=*/builder.createBool(loc, false));
            createCallMemcpy(args);

            builder.create<fir::StoreOp>(loc, plusOne, buffPos);
          }
        },
        [&](const fir::ArrayBoxValue &v) { doAbstractArray(v); },
        [&](const fir::CharArrayBoxValue &v) { doAbstractArray(v); },
        [&](const auto &) {
          TODO(loc, "unhandled array constructor expression");
        });
    return mem;
  }

  // Lower the expr cases in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::Expr<A> &x, mlir::Type,
                          mlir::Value, mlir::Value, mlir::Value,
                          Fortran::lower::StatementContext &stmtCtx) {
    if (isArray(x))
      return {lowerNewArrayExpression(converter, symMap, stmtCtx, toEvExpr(x)),
              /*needCopy=*/true};
    return {asScalar(x), /*needCopy=*/true};
  }

  // Lower an ac-implied-do in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::ImpliedDo<A> &x,
                          mlir::Type resTy, mlir::Value mem,
                          mlir::Value buffPos, mlir::Value buffSize,
                          Fortran::lower::StatementContext &) {
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value lo =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.lower())));
    mlir::Value up =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.upper())));
    mlir::Value step =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.stride())));
    auto seqTy = resTy.template cast<fir::SequenceType>();
    mlir::Type eleTy = fir::unwrapSequenceType(seqTy);
    auto loop =
        builder.create<fir::DoLoopOp>(loc, lo, up, step, /*unordered=*/false,
                                      /*finalCount=*/false, mem);
    // create a new binding for x.name(), to ac-do-variable, to the iteration
    // value.
    symMap.pushImpliedDoBinding(toStringRef(x.name()), loop.getInductionVar());
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loop.getBody());
    // Thread mem inside the loop via loop argument.
    mem = loop.getRegionIterArgs()[0];

    mlir::Type eleRefTy = builder.getRefType(eleTy);

    // Any temps created in the loop body must be freed inside the loop body.
    stmtCtx.pushScope();
    std::optional<mlir::Value> charLen;
    for (const Fortran::evaluate::ArrayConstructorValue<A> &acv : x.values()) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &v) {
            return genArrayCtorInitializer(v, resTy, mem, buffPos, buffSize,
                                           stmtCtx);
          },
          acv.u);
      mlir::Value eleSz = computeElementSize(exv, eleTy, resTy);
      mem = copyNeeded ? copyNextArrayCtorSection(exv, buffPos, buffSize, mem,
                                                  eleSz, eleTy, eleRefTy, resTy)
                       : fir::getBase(exv);
      if (fir::isa_char(seqTy.getEleTy()) && !charLen) {
        charLen = builder.createTemporary(loc, builder.getI64Type());
        mlir::Value castLen =
            builder.createConvert(loc, builder.getI64Type(), fir::getLen(exv));
        assert(charLen.has_value());
        builder.create<fir::StoreOp>(loc, castLen, *charLen);
      }
    }
    stmtCtx.finalizeAndPop();

    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    mem = loop.getResult(0);
    symMap.popImpliedDoBinding();
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos).getResult()};

    // Convert to extended value.
    if (fir::isa_char(seqTy.getEleTy())) {
      assert(charLen.has_value());
      auto len = builder.create<fir::LoadOp>(loc, *charLen);
      return {fir::CharArrayBoxValue{mem, len, extents}, /*needCopy=*/false};
    }
    return {fir::ArrayBoxValue{mem, extents}, /*needCopy=*/false};
  }

  // To simplify the handling and interaction between the various cases, array
  // constructors are always lowered to the incremental construction code
  // pattern, even if the extent of the array value is constant. After the
  // MemToReg pass and constant folding, the optimizer should be able to
  // determine that all the buffer overrun tests are false when the
  // incremental construction wasn't actually required.
  template <typename A>
  CC genarr(const Fortran::evaluate::ArrayConstructor<A> &x) {
    mlir::Location loc = getLoc();
    auto evExpr = toEvExpr(x);
    mlir::Type resTy = translateSomeExprToFIRType(converter, evExpr);
    mlir::IndexType idxTy = builder.getIndexType();
    auto seqTy = resTy.template cast<fir::SequenceType>();
    mlir::Type eleTy = fir::unwrapSequenceType(resTy);
    mlir::Value buffSize = builder.createTemporary(loc, idxTy, ".buff.size");
    mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value buffPos = builder.createTemporary(loc, idxTy, ".buff.pos");
    builder.create<fir::StoreOp>(loc, zero, buffPos);
    // Allocate space for the array to be constructed.
    mlir::Value mem;
    if (fir::hasDynamicSize(resTy)) {
      if (fir::hasDynamicSize(eleTy)) {
        // The size of each element may depend on a general expression. Defer
        // creating the buffer until after the expression is evaluated.
        mem = builder.createNullConstant(loc, builder.getRefType(eleTy));
        builder.create<fir::StoreOp>(loc, zero, buffSize);
      } else {
        mlir::Value initBuffSz =
            builder.createIntegerConstant(loc, idxTy, clInitialBufferSize);
        mem = builder.create<fir::AllocMemOp>(
            loc, eleTy, /*typeparams=*/std::nullopt, initBuffSz);
        builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
      }
    } else {
      mem = builder.create<fir::AllocMemOp>(loc, resTy);
      int64_t buffSz = 1;
      for (auto extent : seqTy.getShape())
        buffSz *= extent;
      mlir::Value initBuffSz =
          builder.createIntegerConstant(loc, idxTy, buffSz);
      builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
    }
    // Compute size of element
    mlir::Type eleRefTy = builder.getRefType(eleTy);

    // Populate the buffer with the elements, growing as necessary.
    std::optional<mlir::Value> charLen;
    for (const auto &expr : x) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &e) {
            return genArrayCtorInitializer(e, resTy, mem, buffPos, buffSize,
                                           stmtCtx);
          },
          expr.u);
      mlir::Value eleSz = computeElementSize(exv, eleTy, resTy);
      mem = copyNeeded ? copyNextArrayCtorSection(exv, buffPos, buffSize, mem,
                                                  eleSz, eleTy, eleRefTy, resTy)
                       : fir::getBase(exv);
      if (fir::isa_char(seqTy.getEleTy()) && !charLen) {
        charLen = builder.createTemporary(loc, builder.getI64Type());
        mlir::Value castLen =
            builder.createConvert(loc, builder.getI64Type(), fir::getLen(exv));
        builder.create<fir::StoreOp>(loc, castLen, *charLen);
      }
    }
    mem = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos)};

    // Cleanup the temporary.
    fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup(
        [bldr, loc, mem]() { bldr->create<fir::FreeMemOp>(loc, mem); });

    // Return the continuation.
    if (fir::isa_char(seqTy.getEleTy())) {
      if (charLen) {
        auto len = builder.create<fir::LoadOp>(loc, *charLen);
        return genarr(fir::CharArrayBoxValue{mem, len, extents});
      }
      return genarr(fir::CharArrayBoxValue{mem, zero, extents});
    }
    return genarr(fir::ArrayBoxValue{mem, extents});
  }

  CC genarr(const Fortran::evaluate::ImpliedDoIndex &) {
    fir::emitFatalError(getLoc(), "implied do index cannot have rank > 0");
  }
  CC genarr(const Fortran::evaluate::TypeParamInquiry &x) {
    TODO(getLoc(), "array expr type parameter inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::DescriptorInquiry &x) {
    TODO(getLoc(), "array expr descriptor inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::StructureConstructor &x) {
    TODO(getLoc(), "structure constructor");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }

  //===--------------------------------------------------------------------===//
  // LOCICAL operators (.NOT., .AND., .EQV., etc.)
  //===--------------------------------------------------------------------===//

  template <int KIND>
  CC genarr(const Fortran::evaluate::Not<KIND> &x) {
    mlir::Location loc = getLoc();
    mlir::IntegerType i1Ty = builder.getI1Type();
    auto lambda = genarr(x.left());
    mlir::Value truth = builder.createBool(loc, true);
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value logical = fir::getBase(lambda(iters));
      mlir::Value val = builder.createConvert(loc, i1Ty, logical);
      return builder.create<mlir::arith::XOrIOp>(loc, val, truth);
    };
  }
  template <typename OP, typename A>
  CC createBinaryBoolOp(const A &x) {
    mlir::Location loc = getLoc();
    mlir::IntegerType i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value left = fir::getBase(lf(iters));
      mlir::Value right = fir::getBase(rf(iters));
      mlir::Value lhs = builder.createConvert(loc, i1Ty, left);
      mlir::Value rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, lhs, rhs);
    };
  }
  template <typename OP, typename A>
  CC createCompareBoolOp(mlir::arith::CmpIPredicate pred, const A &x) {
    mlir::Location loc = getLoc();
    mlir::IntegerType i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value left = fir::getBase(lf(iters));
      mlir::Value right = fir::getBase(rf(iters));
      mlir::Value lhs = builder.createConvert(loc, i1Ty, left);
      mlir::Value rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::LogicalOperation<KIND> &x) {
    switch (x.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryBoolOp<mlir::arith::AndIOp>(x);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryBoolOp<mlir::arith::OrIOp>(x);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareBoolOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::eq, x);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareBoolOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::ne, x);
    case Fortran::evaluate::LogicalOperator::Not:
      llvm_unreachable(".NOT. handled elsewhere");
    }
    llvm_unreachable("unhandled case");
  }

  //===--------------------------------------------------------------------===//
  // Relational operators (<, <=, ==, etc.)
  //===--------------------------------------------------------------------===//

  template <typename OP, typename PRED, typename A>
  CC createCompareOp(PRED pred, const A &x) {
    mlir::Location loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      mlir::Value lhs = fir::getBase(lf(iters));
      mlir::Value rhs = fir::getBase(rf(iters));
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <typename A>
  CC createCompareCharOp(mlir::arith::CmpIPredicate pred, const A &x) {
    mlir::Location loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      return fir::runtime::genCharCompare(builder, loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    return createCompareOp<mlir::arith::CmpIOp>(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Character, KIND>> &x) {
    return createCompareCharOp(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    return createCompareOp<mlir::arith::CmpFOp>(translateFloatRelational(x.opr),
                                                x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    return createCompareOp<fir::CmpcOp>(translateFloatRelational(x.opr), x);
  }
  CC genarr(
      const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &r) {
    return std::visit([&](const auto &x) { return genarr(x); }, r.u);
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Designator<A> &des) {
    ComponentPath components(des.Rank() > 0);
    return std::visit([&](const auto &x) { return genarr(x, components); },
                      des.u);
  }

  /// Is the path component rank > 0?
  static bool ranked(const PathComponent &x) {
    return std::visit(Fortran::common::visitors{
                          [](const ImplicitSubscripts &) { return false; },
                          [](const auto *v) { return v->Rank() > 0; }},
                      x);
  }

  void extendComponent(Fortran::lower::ComponentPath &component,
                       mlir::Type coorTy, mlir::ValueRange vals) {
    auto *bldr = &converter.getFirOpBuilder();
    llvm::SmallVector<mlir::Value> offsets(vals.begin(), vals.end());
    auto currentFunc = component.getExtendCoorRef();
    auto loc = getLoc();
    auto newCoorRef = [bldr, coorTy, offsets, currentFunc,
                       loc](mlir::Value val) -> mlir::Value {
      return bldr->create<fir::CoordinateOp>(loc, bldr->getRefType(coorTy),
                                             currentFunc(val), offsets);
    };
    component.extendCoorRef = newCoorRef;
  }

  //===-------------------------------------------------------------------===//
  // Array data references in an explicit iteration space.
  //
  // Use the base array that was loaded before the loop nest.
  //===-------------------------------------------------------------------===//

  /// Lower the path (`revPath`, in reverse) to be appended to an array_fetch or
  /// array_update op. \p ty is the initial type of the array
  /// (reference). Returns the type of the element after application of the
  /// path in \p components.
  ///
  /// TODO: This needs to deal with array's with initial bounds other than 1.
  /// TODO: Thread type parameters correctly.
  mlir::Type lowerPath(const ExtValue &arrayExv, ComponentPath &components) {
    mlir::Location loc = getLoc();
    mlir::Type ty = fir::getBase(arrayExv).getType();
    auto &revPath = components.reversePath;
    ty = fir::unwrapPassByRefType(ty);
    bool prefix = true;
    bool deref = false;
    auto addComponentList = [&](mlir::Type ty, mlir::ValueRange vals) {
      if (deref) {
        extendComponent(components, ty, vals);
      } else if (prefix) {
        for (auto v : vals)
          components.prefixComponents.push_back(v);
      } else {
        for (auto v : vals)
          components.suffixComponents.push_back(v);
      }
    };
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    bool atBase = true;
    auto saveSemant = semant;
    if (isProjectedCopyInCopyOut())
      semant = ConstituentSemantics::RefTransparent;
    unsigned index = 0;
    for (const auto &v : llvm::reverse(revPath)) {
      std::visit(
          Fortran::common::visitors{
              [&](const ImplicitSubscripts &) {
                prefix = false;
                ty = fir::unwrapSequenceType(ty);
              },
              [&](const Fortran::evaluate::ComplexPart *x) {
                assert(!prefix && "complex part must be at end");
                mlir::Value offset = builder.createIntegerConstant(
                    loc, builder.getI32Type(),
                    x->part() == Fortran::evaluate::ComplexPart::Part::RE ? 0
                                                                          : 1);
                components.suffixComponents.push_back(offset);
                ty = fir::applyPathToType(ty, mlir::ValueRange{offset});
              },
              [&](const Fortran::evaluate::ArrayRef *x) {
                if (Fortran::lower::isRankedArrayAccess(*x)) {
                  genSliceIndices(components, arrayExv, *x, atBase);
                  ty = fir::unwrapSeqOrBoxedSeqType(ty);
                } else {
                  // Array access where the expressions are scalar and cannot
                  // depend upon the implied iteration space.
                  unsigned ssIndex = 0u;
                  llvm::SmallVector<mlir::Value> componentsToAdd;
                  for (const auto &ss : x->subscript()) {
                    std::visit(
                        Fortran::common::visitors{
                            [&](const Fortran::evaluate::
                                    IndirectSubscriptIntegerExpr &ie) {
                              const auto &e = ie.value();
                              if (isArray(e))
                                fir::emitFatalError(
                                    loc,
                                    "multiple components along single path "
                                    "generating array subexpressions");
                              // Lower scalar index expression, append it to
                              // subs.
                              mlir::Value subscriptVal =
                                  fir::getBase(asScalarArray(e));
                              // arrayExv is the base array. It needs to reflect
                              // the current array component instead.
                              // FIXME: must use lower bound of this component,
                              // not just the constant 1.
                              mlir::Value lb =
                                  atBase ? fir::factory::readLowerBound(
                                               builder, loc, arrayExv, ssIndex,
                                               one)
                                         : one;
                              mlir::Value val = builder.createConvert(
                                  loc, idxTy, subscriptVal);
                              mlir::Value ivAdj =
                                  builder.create<mlir::arith::SubIOp>(
                                      loc, idxTy, val, lb);
                              componentsToAdd.push_back(
                                  builder.createConvert(loc, idxTy, ivAdj));
                            },
                            [&](const auto &) {
                              fir::emitFatalError(
                                  loc, "multiple components along single path "
                                       "generating array subexpressions");
                            }},
                        ss.u);
                    ssIndex++;
                  }
                  ty = fir::unwrapSeqOrBoxedSeqType(ty);
                  addComponentList(ty, componentsToAdd);
                }
              },
              [&](const Fortran::evaluate::Component *x) {
                auto fieldTy = fir::FieldType::get(builder.getContext());
                llvm::StringRef name = toStringRef(getLastSym(*x).name());
                if (auto recTy = ty.dyn_cast<fir::RecordType>()) {
                  ty = recTy.getType(name);
                  auto fld = builder.create<fir::FieldIndexOp>(
                      loc, fieldTy, name, recTy, fir::getTypeParams(arrayExv));
                  addComponentList(ty, {fld});
                  if (index != revPath.size() - 1 || !isPointerAssignment()) {
                    // Need an intermediate  dereference if the boxed value
                    // appears in the middle of the component path or if it is
                    // on the right and this is not a pointer assignment.
                    if (auto boxTy = ty.dyn_cast<fir::BaseBoxType>()) {
                      auto currentFunc = components.getExtendCoorRef();
                      auto loc = getLoc();
                      auto *bldr = &converter.getFirOpBuilder();
                      auto newCoorRef = [=](mlir::Value val) -> mlir::Value {
                        return bldr->create<fir::LoadOp>(loc, currentFunc(val));
                      };
                      components.extendCoorRef = newCoorRef;
                      deref = true;
                    }
                  }
                } else if (auto boxTy = ty.dyn_cast<fir::BaseBoxType>()) {
                  ty = fir::unwrapRefType(boxTy.getEleTy());
                  auto recTy = ty.cast<fir::RecordType>();
                  ty = recTy.getType(name);
                  auto fld = builder.create<fir::FieldIndexOp>(
                      loc, fieldTy, name, recTy, fir::getTypeParams(arrayExv));
                  extendComponent(components, ty, {fld});
                } else {
                  TODO(loc, "other component type");
                }
              }},
          v);
      atBase = false;
      ++index;
    }
    semant = saveSemant;
    ty = fir::unwrapSequenceType(ty);
    components.applied = true;
    return ty;
  }

  llvm::SmallVector<mlir::Value> genSubstringBounds(ComponentPath &components) {
    llvm::SmallVector<mlir::Value> result;
    if (components.substring)
      populateBounds(result, components.substring);
    return result;
  }

  CC applyPathToArrayLoad(fir::ArrayLoadOp load, ComponentPath &components) {
    mlir::Location loc = getLoc();
    auto revPath = components.reversePath;
    fir::ExtendedValue arrayExv =
        arrayLoadExtValue(builder, loc, load, {}, load);
    mlir::Type eleTy = lowerPath(arrayExv, components);
    auto currentPC = components.pc;
    auto pc = [=, prefix = components.prefixComponents,
               suffix = components.suffixComponents](IterSpace iters) {
      // Add path prefix and suffix.
      return IterationSpace(currentPC(iters), prefix, suffix);
    };
    components.resetPC();
    llvm::SmallVector<mlir::Value> substringBounds =
        genSubstringBounds(components);
    if (isProjectedCopyInCopyOut()) {
      destination = load;
      auto lambda = [=, esp = this->explicitSpace](IterSpace iters) mutable {
        mlir::Value innerArg = esp->findArgumentOfLoad(load);
        if (isAdjustedArrayElementType(eleTy)) {
          mlir::Type eleRefTy = builder.getRefType(eleTy);
          auto arrayOp = builder.create<fir::ArrayAccessOp>(
              loc, eleRefTy, innerArg, iters.iterVec(),
              fir::factory::getTypeParams(loc, builder, load));
          if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
            mlir::Value dstLen = fir::factory::genLenOfCharacter(
                builder, loc, load, iters.iterVec(), substringBounds);
            fir::ArrayAmendOp amend = createCharArrayAmend(
                loc, builder, arrayOp, dstLen, iters.elementExv(), innerArg,
                substringBounds);
            return arrayLoadExtValue(builder, loc, load, iters.iterVec(), amend,
                                     dstLen);
          }
          if (fir::isa_derived(eleTy)) {
            fir::ArrayAmendOp amend =
                createDerivedArrayAmend(loc, load, builder, arrayOp,
                                        iters.elementExv(), eleTy, innerArg);
            return arrayLoadExtValue(builder, loc, load, iters.iterVec(),
                                     amend);
          }
          assert(eleTy.isa<fir::SequenceType>());
          TODO(loc, "array (as element) assignment");
        }
        if (components.hasExtendCoorRef()) {
          auto eleBoxTy =
              fir::applyPathToType(innerArg.getType(), iters.iterVec());
          if (!eleBoxTy || !eleBoxTy.isa<fir::BoxType>())
            TODO(loc, "assignment in a FORALL involving a designator with a "
                      "POINTER or ALLOCATABLE component part-ref");
          auto arrayOp = builder.create<fir::ArrayAccessOp>(
              loc, builder.getRefType(eleBoxTy), innerArg, iters.iterVec(),
              fir::factory::getTypeParams(loc, builder, load));
          mlir::Value addr = components.getExtendCoorRef()(arrayOp);
          components.resetExtendCoorRef();
          // When the lhs is a boxed value and the context is not a pointer
          // assignment, then insert the dereference of the box before any
          // conversion and store.
          if (!isPointerAssignment()) {
            if (auto boxTy = eleTy.dyn_cast<fir::BaseBoxType>()) {
              eleTy = fir::boxMemRefType(boxTy);
              addr = builder.create<fir::BoxAddrOp>(loc, eleTy, addr);
              eleTy = fir::unwrapRefType(eleTy);
            }
          }
          auto ele = convertElementForUpdate(loc, eleTy, iters.getElement());
          builder.create<fir::StoreOp>(loc, ele, addr);
          auto amend = builder.create<fir::ArrayAmendOp>(
              loc, innerArg.getType(), innerArg, arrayOp);
          return arrayLoadExtValue(builder, loc, load, iters.iterVec(), amend);
        }
        auto ele = convertElementForUpdate(loc, eleTy, iters.getElement());
        auto update = builder.create<fir::ArrayUpdateOp>(
            loc, innerArg.getType(), innerArg, ele, iters.iterVec(),
            fir::factory::getTypeParams(loc, builder, load));
        return arrayLoadExtValue(builder, loc, load, iters.iterVec(), update);
      };
      return [=](IterSpace iters) mutable { return lambda(pc(iters)); };
    }
    if (isCustomCopyInCopyOut()) {
      // Create an array_modify to get the LHS element address and indicate
      // the assignment, and create the call to the user defined assignment.
      destination = load;
      auto lambda = [=](IterSpace iters) mutable {
        mlir::Value innerArg = explicitSpace->findArgumentOfLoad(load);
        mlir::Type refEleTy =
            fir::isa_ref_type(eleTy) ? eleTy : builder.getRefType(eleTy);
        auto arrModify = builder.create<fir::ArrayModifyOp>(
            loc, mlir::TypeRange{refEleTy, innerArg.getType()}, innerArg,
            iters.iterVec(), load.getTypeparams());
        return arrayLoadExtValue(builder, loc, load, iters.iterVec(),
                                 arrModify.getResult(1));
      };
      return [=](IterSpace iters) mutable { return lambda(pc(iters)); };
    }
    auto lambda = [=, semant = this->semant](IterSpace iters) mutable {
      if (semant == ConstituentSemantics::RefOpaque ||
          isAdjustedArrayElementType(eleTy)) {
        mlir::Type resTy = builder.getRefType(eleTy);
        // Use array element reference semantics.
        auto access = builder.create<fir::ArrayAccessOp>(
            loc, resTy, load, iters.iterVec(),
            fir::factory::getTypeParams(loc, builder, load));
        mlir::Value newBase = access;
        if (fir::isa_char(eleTy)) {
          mlir::Value dstLen = fir::factory::genLenOfCharacter(
              builder, loc, load, iters.iterVec(), substringBounds);
          if (!substringBounds.empty()) {
            fir::CharBoxValue charDst{access, dstLen};
            fir::factory::CharacterExprHelper helper{builder, loc};
            charDst = helper.createSubstring(charDst, substringBounds);
            newBase = charDst.getAddr();
          }
          return arrayLoadExtValue(builder, loc, load, iters.iterVec(), newBase,
                                   dstLen);
        }
        return arrayLoadExtValue(builder, loc, load, iters.iterVec(), newBase);
      }
      if (components.hasExtendCoorRef()) {
        auto eleBoxTy = fir::applyPathToType(load.getType(), iters.iterVec());
        if (!eleBoxTy || !eleBoxTy.isa<fir::BoxType>())
          TODO(loc, "assignment in a FORALL involving a designator with a "
                    "POINTER or ALLOCATABLE component part-ref");
        auto access = builder.create<fir::ArrayAccessOp>(
            loc, builder.getRefType(eleBoxTy), load, iters.iterVec(),
            fir::factory::getTypeParams(loc, builder, load));
        mlir::Value addr = components.getExtendCoorRef()(access);
        components.resetExtendCoorRef();
        return arrayLoadExtValue(builder, loc, load, iters.iterVec(), addr);
      }
      if (isPointerAssignment()) {
        auto eleTy = fir::applyPathToType(load.getType(), iters.iterVec());
        if (!eleTy.isa<fir::BoxType>()) {
          // Rhs is a regular expression that will need to be boxed before
          // assigning to the boxed variable.
          auto typeParams = fir::factory::getTypeParams(loc, builder, load);
          auto access = builder.create<fir::ArrayAccessOp>(
              loc, builder.getRefType(eleTy), load, iters.iterVec(),
              typeParams);
          auto addr = components.getExtendCoorRef()(access);
          components.resetExtendCoorRef();
          auto ptrEleTy = fir::PointerType::get(eleTy);
          auto ptrAddr = builder.createConvert(loc, ptrEleTy, addr);
          auto boxTy = fir::BoxType::get(ptrEleTy);
          // FIXME: The typeparams to the load may be different than those of
          // the subobject.
          if (components.hasExtendCoorRef())
            TODO(loc, "need to adjust typeparameter(s) to reflect the final "
                      "component");
          mlir::Value embox =
              builder.create<fir::EmboxOp>(loc, boxTy, ptrAddr,
                                           /*shape=*/mlir::Value{},
                                           /*slice=*/mlir::Value{}, typeParams);
          return arrayLoadExtValue(builder, loc, load, iters.iterVec(), embox);
        }
      }
      auto fetch = builder.create<fir::ArrayFetchOp>(
          loc, eleTy, load, iters.iterVec(), load.getTypeparams());
      return arrayLoadExtValue(builder, loc, load, iters.iterVec(), fetch);
    };
    return [=](IterSpace iters) mutable { return lambda(pc(iters)); };
  }

  template <typename A>
  CC genImplicitArrayAccess(const A &x, ComponentPath &components) {
    components.reversePath.push_back(ImplicitSubscripts{});
    ExtValue exv = asScalarRef(x);
    lowerPath(exv, components);
    auto lambda = genarr(exv, components);
    return [=](IterSpace iters) { return lambda(components.pc(iters)); };
  }
  CC genImplicitArrayAccess(const Fortran::evaluate::NamedEntity &x,
                            ComponentPath &components) {
    if (x.IsSymbol())
      return genImplicitArrayAccess(getFirstSym(x), components);
    return genImplicitArrayAccess(x.GetComponent(), components);
  }

  template <typename A>
  CC genAsScalar(const A &x) {
    mlir::Location loc = getLoc();
    if (isProjectedCopyInCopyOut()) {
      return [=, &x, builder = &converter.getFirOpBuilder()](
                 IterSpace iters) -> ExtValue {
        ExtValue exv = asScalarRef(x);
        mlir::Value addr = fir::getBase(exv);
        mlir::Type eleTy = fir::unwrapRefType(addr.getType());
        if (isAdjustedArrayElementType(eleTy)) {
          if (fir::isa_char(eleTy)) {
            fir::factory::CharacterExprHelper{*builder, loc}.createAssign(
                exv, iters.elementExv());
          } else if (fir::isa_derived(eleTy)) {
            TODO(loc, "assignment of derived type");
          } else {
            fir::emitFatalError(loc, "array type not expected in scalar");
          }
        } else {
          auto eleVal = convertElementForUpdate(loc, eleTy, iters.getElement());
          builder->create<fir::StoreOp>(loc, eleVal, addr);
        }
        return exv;
      };
    }
    return [=, &x](IterSpace) { return asScalar(x); };
  }

  bool tailIsPointerInPointerAssignment(const Fortran::semantics::Symbol &x,
                                        ComponentPath &components) {
    return isPointerAssignment() && Fortran::semantics::IsPointer(x) &&
           !components.hasComponents();
  }
  bool tailIsPointerInPointerAssignment(const Fortran::evaluate::Component &x,
                                        ComponentPath &components) {
    return tailIsPointerInPointerAssignment(getLastSym(x), components);
  }

  CC genarr(const Fortran::semantics::Symbol &x, ComponentPath &components) {
    if (explicitSpaceIsActive()) {
      if (x.Rank() > 0 && !tailIsPointerInPointerAssignment(x, components))
        components.reversePath.push_back(ImplicitSubscripts{});
      if (fir::ArrayLoadOp load = explicitSpace->findBinding(&x))
        return applyPathToArrayLoad(load, components);
    } else {
      return genImplicitArrayAccess(x, components);
    }
    if (pathIsEmpty(components))
      return components.substring ? genAsScalar(*components.substring)
                                  : genAsScalar(x);
    mlir::Location loc = getLoc();
    return [=](IterSpace) -> ExtValue {
      fir::emitFatalError(loc, "reached symbol with path");
    };
  }

  /// Lower a component path with or without rank.
  /// Example: <code>array%baz%qux%waldo</code>
  CC genarr(const Fortran::evaluate::Component &x, ComponentPath &components) {
    if (explicitSpaceIsActive()) {
      if (x.base().Rank() == 0 && x.Rank() > 0 &&
          !tailIsPointerInPointerAssignment(x, components))
        components.reversePath.push_back(ImplicitSubscripts{});
      if (fir::ArrayLoadOp load = explicitSpace->findBinding(&x))
        return applyPathToArrayLoad(load, components);
    } else {
      if (x.base().Rank() == 0)
        return genImplicitArrayAccess(x, components);
    }
    bool atEnd = pathIsEmpty(components);
    if (!getLastSym(x).test(Fortran::semantics::Symbol::Flag::ParentComp))
      // Skip parent components; their components are placed directly in the
      // object.
      components.reversePath.push_back(&x);
    auto result = genarr(x.base(), components);
    if (components.applied)
      return result;
    if (atEnd)
      return genAsScalar(x);
    mlir::Location loc = getLoc();
    return [=](IterSpace) -> ExtValue {
      fir::emitFatalError(loc, "reached component with path");
    };
  }

  /// Array reference with subscripts. If this has rank > 0, this is a form
  /// of an array section (slice).
  ///
  /// There are two "slicing" primitives that may be applied on a dimension by
  /// dimension basis: (1) triple notation and (2) vector addressing. Since
  /// dimensions can be selectively sliced, some dimensions may contain
  /// regular scalar expressions and those dimensions do not participate in
  /// the array expression evaluation.
  CC genarr(const Fortran::evaluate::ArrayRef &x, ComponentPath &components) {
    if (explicitSpaceIsActive()) {
      if (Fortran::lower::isRankedArrayAccess(x))
        components.reversePath.push_back(ImplicitSubscripts{});
      if (fir::ArrayLoadOp load = explicitSpace->findBinding(&x)) {
        components.reversePath.push_back(&x);
        return applyPathToArrayLoad(load, components);
      }
    } else {
      if (Fortran::lower::isRankedArrayAccess(x)) {
        components.reversePath.push_back(&x);
        return genImplicitArrayAccess(x.base(), components);
      }
    }
    bool atEnd = pathIsEmpty(components);
    components.reversePath.push_back(&x);
    auto result = genarr(x.base(), components);
    if (components.applied)
      return result;
    mlir::Location loc = getLoc();
    if (atEnd) {
      if (x.Rank() == 0)
        return genAsScalar(x);
      fir::emitFatalError(loc, "expected scalar");
    }
    return [=](IterSpace) -> ExtValue {
      fir::emitFatalError(loc, "reached arrayref with path");
    };
  }

  CC genarr(const Fortran::evaluate::CoarrayRef &x, ComponentPath &components) {
    TODO(getLoc(), "coarray reference");
  }

  CC genarr(const Fortran::evaluate::NamedEntity &x,
            ComponentPath &components) {
    return x.IsSymbol() ? genarr(getFirstSym(x), components)
                        : genarr(x.GetComponent(), components);
  }

  CC genarr(const Fortran::evaluate::DataRef &x, ComponentPath &components) {
    return std::visit([&](const auto &v) { return genarr(v, components); },
                      x.u);
  }

  bool pathIsEmpty(const ComponentPath &components) {
    return components.reversePath.empty();
  }

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, semant{sem} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem,
                             Fortran::lower::ExplicitIterSpace *expSpace,
                             Fortran::lower::ImplicitIterSpace *impSpace)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap},
        explicitSpace((expSpace && expSpace->isActive()) ? expSpace : nullptr),
        implicitSpace((impSpace && !impSpace->empty()) ? impSpace : nullptr),
        semant{sem} {
    // Generate any mask expressions, as necessary. This is the compute step
    // that creates the effective masks. See 10.2.3.2 in particular.
    genMasks();
  }

  mlir::Location getLoc() { return converter.getCurrentLocation(); }

  /// Array appears in a lhs context such that it is assigned after the rhs is
  /// fully evaluated.
  inline bool isCopyInCopyOut() {
    return semant == ConstituentSemantics::CopyInCopyOut;
  }

  /// Array appears in a lhs (or temp) context such that a projected,
  /// discontiguous subspace of the array is assigned after the rhs is fully
  /// evaluated. That is, the rhs array value is merged into a section of the
  /// lhs array.
  inline bool isProjectedCopyInCopyOut() {
    return semant == ConstituentSemantics::ProjectedCopyInCopyOut;
  }

  // ???: Do we still need this?
  inline bool isCustomCopyInCopyOut() {
    return semant == ConstituentSemantics::CustomCopyInCopyOut;
  }

  /// Are we lowering in a left-hand side context?
  inline bool isLeftHandSide() {
    return isCopyInCopyOut() || isProjectedCopyInCopyOut() ||
           isCustomCopyInCopyOut();
  }

  /// Array appears in a context where it must be boxed.
  inline bool isBoxValue() { return semant == ConstituentSemantics::BoxValue; }

  /// Array appears in a context where differences in the memory reference can
  /// be observable in the computational results. For example, an array
  /// element is passed to an impure procedure.
  inline bool isReferentiallyOpaque() {
    return semant == ConstituentSemantics::RefOpaque;
  }

  /// Array appears in a context where it is passed as a VALUE argument.
  inline bool isValueAttribute() {
    return semant == ConstituentSemantics::ByValueArg;
  }

  /// Can the loops over the expression be unordered?
  inline bool isUnordered() const { return unordered; }

  void setUnordered(bool b) { unordered = b; }

  inline bool isPointerAssignment() const { return lbounds.has_value(); }

  inline bool isBoundsSpec() const {
    return isPointerAssignment() && !ubounds.has_value();
  }

  inline bool isBoundsRemap() const {
    return isPointerAssignment() && ubounds.has_value();
  }

  void setPointerAssignmentBounds(
      const llvm::SmallVector<mlir::Value> &lbs,
      std::optional<llvm::SmallVector<mlir::Value>> ubs) {
    lbounds = lbs;
    ubounds = ubs;
  }

  void setLoweredProcRef(const Fortran::evaluate::ProcedureRef *procRef) {
    loweredProcRef = procRef;
  }

  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  bool elementCtx = false;
  Fortran::lower::SymMap &symMap;
  /// The continuation to generate code to update the destination.
  std::optional<CC> ccStoreToDest;
  std::optional<std::function<void(llvm::ArrayRef<mlir::Value>)>> ccPrelude;
  std::optional<std::function<fir::ArrayLoadOp(llvm::ArrayRef<mlir::Value>)>>
      ccLoadDest;
  /// The destination is the loaded array into which the results will be
  /// merged.
  fir::ArrayLoadOp destination;
  /// The shape of the destination.
  llvm::SmallVector<mlir::Value> destShape;
  /// List of arrays in the expression that have been loaded.
  llvm::SmallVector<ArrayOperand> arrayOperands;
  /// If there is a user-defined iteration space, explicitShape will hold the
  /// information from the front end.
  Fortran::lower::ExplicitIterSpace *explicitSpace = nullptr;
  Fortran::lower::ImplicitIterSpace *implicitSpace = nullptr;
  ConstituentSemantics semant = ConstituentSemantics::RefTransparent;
  /// `lbounds`, `ubounds` are used in POINTER value assignments, which may only
  /// occur in an explicit iteration space.
  std::optional<llvm::SmallVector<mlir::Value>> lbounds;
  std::optional<llvm::SmallVector<mlir::Value>> ubounds;
  // Can the array expression be evaluated in any order?
  // Will be set to false if any of the expression parts prevent this.
  bool unordered = true;
  // ProcedureRef currently being lowered. Used to retrieve the iteration shape
  // in elemental context with passed object.
  const Fortran::evaluate::ProcedureRef *loweredProcRef = nullptr;
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeInitializerExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx,
                            /*inInitializer=*/true}
      .genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering(loc, converter, symMap, stmtCtx).gen(expr);
}

fir::ExtendedValue Fortran::lower::createInitializerAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering(loc, converter, symMap, stmtCtx,
                            /*inInitializer=*/true)
      .gen(expr);
}

void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ") << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}

void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter, const fir::ExtendedValue &lhs,
    const Fortran::lower::SomeExpr &rhs, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "onto array: " << lhs << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ") << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}
void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter, const fir::ExtendedValue &lhs,
    const fir::ExtendedValue &rhs, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "onto array: " << lhs << '\n';
             llvm::dbgs() << "assign expression: " << rhs << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}

void Fortran::lower::createAnyMaskedArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
    Fortran::lower::ExplicitIterSpace &explicitSpace,
    Fortran::lower::ImplicitIterSpace &implicitSpace,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given the explicit iteration space:\n"
             << explicitSpace << "\n and implied mask conditions:\n"
             << implicitSpace << '\n';);
  ArrayExprLowering::lowerAnyMaskedArrayAssignment(
      converter, symMap, stmtCtx, lhs, rhs, explicitSpace, implicitSpace);
}

void Fortran::lower::createAllocatableArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
    Fortran::lower::ExplicitIterSpace &explicitSpace,
    Fortran::lower::ImplicitIterSpace &implicitSpace,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "defining array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given the explicit iteration space:\n"
             << explicitSpace << "\n and implied mask conditions:\n"
             << implicitSpace << '\n';);
  ArrayExprLowering::lowerAllocatableArrayAssignment(
      converter, symMap, stmtCtx, lhs, rhs, explicitSpace, implicitSpace);
}

void Fortran::lower::createArrayOfPointerAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &lhs, const Fortran::lower::SomeExpr &rhs,
    Fortran::lower::ExplicitIterSpace &explicitSpace,
    Fortran::lower::ImplicitIterSpace &implicitSpace,
    const llvm::SmallVector<mlir::Value> &lbounds,
    std::optional<llvm::SmallVector<mlir::Value>> ubounds,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "defining pointer: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given the explicit iteration space:\n"
             << explicitSpace << "\n and implied mask conditions:\n"
             << implicitSpace << '\n';);
  assert(explicitSpace.isActive() && "must be in FORALL construct");
  ArrayExprLowering::lowerArrayOfPointerAssignment(
      converter, symMap, stmtCtx, lhs, rhs, explicitSpace, implicitSpace,
      lbounds, ubounds);
}

fir::ExtendedValue Fortran::lower::createSomeArrayTempValue(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  return ArrayExprLowering::lowerNewArrayExpression(converter, symMap, stmtCtx,
                                                    expr);
}

void Fortran::lower::createLazyArrayTempValue(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, mlir::Value raggedHeader,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  ArrayExprLowering::lowerLazyArrayExpression(converter, symMap, stmtCtx, expr,
                                              raggedHeader);
}

fir::ExtendedValue
Fortran::lower::createSomeArrayBox(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::lower::SomeExpr &expr,
                                   Fortran::lower::SymMap &symMap,
                                   Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "box designator: ") << '\n');
  return ArrayExprLowering::lowerBoxedArrayExpression(converter, symMap,
                                                      stmtCtx, expr);
}

fir::MutableBoxValue Fortran::lower::createMutableBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap) {
  // MutableBox lowering StatementContext does not need to be propagated
  // to the caller because the result value is a variable, not a temporary
  // expression. The StatementContext clean-up can occur before using the
  // resulting MutableBoxValue. Variables of all other types are handled in the
  // bridge.
  Fortran::lower::StatementContext dummyStmtCtx;
  return ScalarExprLowering{loc, converter, symMap, dummyStmtCtx}
      .genMutableBoxValue(expr);
}

bool Fortran::lower::isParentComponent(const Fortran::lower::SomeExpr &expr) {
  if (const Fortran::semantics::Symbol * symbol{GetLastSymbol(expr)}) {
    if (symbol->test(Fortran::semantics::Symbol::Flag::ParentComp))
      return true;
  }
  return false;
}

// Handling special case where the last component is referring to the
// parent component.
//
// TYPE t
//   integer :: a
// END TYPE
// TYPE, EXTENDS(t) :: t2
//   integer :: b
// END TYPE
// TYPE(t2) :: y(2)
// TYPE(t2) :: a
// y(:)%t  ! just need to update the box with a slice pointing to the first
//         ! component of `t`.
// a%t     ! simple conversion to TYPE(t).
fir::ExtendedValue Fortran::lower::updateBoxForParentComponent(
    Fortran::lower::AbstractConverter &converter, fir::ExtendedValue box,
    const Fortran::lower::SomeExpr &expr) {
  mlir::Location loc = converter.getCurrentLocation();
  auto &builder = converter.getFirOpBuilder();
  mlir::Value boxBase = fir::getBase(box);
  mlir::Operation *op = boxBase.getDefiningOp();
  mlir::Type actualTy = converter.genType(expr);

  if (op) {
    if (auto embox = mlir::dyn_cast<fir::EmboxOp>(op)) {
      auto newBox = builder.create<fir::EmboxOp>(
          loc, fir::BoxType::get(actualTy), embox.getMemref(), embox.getShape(),
          embox.getSlice(), embox.getTypeparams());
      return fir::substBase(box, newBox);
    }
    if (auto rebox = mlir::dyn_cast<fir::ReboxOp>(op)) {
      auto newBox = builder.create<fir::ReboxOp>(
          loc, fir::BoxType::get(actualTy), rebox.getBox(), rebox.getShape(),
          rebox.getSlice());
      return fir::substBase(box, newBox);
    }
  }

  mlir::Value empty;
  mlir::ValueRange emptyRange;
  return builder.create<fir::ReboxOp>(loc, fir::BoxType::get(actualTy), boxBase,
                                      /*shape=*/empty,
                                      /*slice=*/empty);
}

fir::ExtendedValue Fortran::lower::createBoxValue(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  if (expr.Rank() > 0 && Fortran::evaluate::IsVariable(expr) &&
      !Fortran::evaluate::HasVectorSubscript(expr)) {
    fir::ExtendedValue result =
        Fortran::lower::createSomeArrayBox(converter, expr, symMap, stmtCtx);
    if (isParentComponent(expr))
      result = updateBoxForParentComponent(converter, result, expr);
    return result;
  }
  fir::ExtendedValue addr = Fortran::lower::createSomeExtendedAddress(
      loc, converter, expr, symMap, stmtCtx);
  fir::ExtendedValue result = fir::BoxValue(
      converter.getFirOpBuilder().createBox(loc, addr, addr.isPolymorphic()));
  if (isParentComponent(expr))
    result = updateBoxForParentComponent(converter, result, expr);
  return result;
}

mlir::Value Fortran::lower::createSubroutineCall(
    AbstractConverter &converter, const evaluate::ProcedureRef &call,
    ExplicitIterSpace &explicitIterSpace, ImplicitIterSpace &implicitIterSpace,
    SymMap &symMap, StatementContext &stmtCtx, bool isUserDefAssignment) {
  mlir::Location loc = converter.getCurrentLocation();

  if (isUserDefAssignment) {
    assert(call.arguments().size() == 2);
    const auto *lhs = call.arguments()[0].value().UnwrapExpr();
    const auto *rhs = call.arguments()[1].value().UnwrapExpr();
    assert(lhs && rhs &&
           "user defined assignment arguments must be expressions");
    if (call.IsElemental() && lhs->Rank() > 0) {
      // Elemental user defined assignment has special requirements to deal with
      // LHS/RHS overlaps. See 10.2.1.5 p2.
      ArrayExprLowering::lowerElementalUserAssignment(
          converter, symMap, stmtCtx, explicitIterSpace, implicitIterSpace,
          call);
    } else if (explicitIterSpace.isActive() && lhs->Rank() == 0) {
      // Scalar defined assignment (elemental or not) in a FORALL context.
      mlir::func::FuncOp func =
          Fortran::lower::CallerInterface(call, converter).getFuncOp();
      ArrayExprLowering::lowerScalarUserAssignment(
          converter, symMap, stmtCtx, explicitIterSpace, func, *lhs, *rhs);
    } else if (explicitIterSpace.isActive()) {
      // TODO: need to array fetch/modify sub-arrays?
      TODO(loc, "non elemental user defined array assignment inside FORALL");
    } else {
      if (!implicitIterSpace.empty())
        fir::emitFatalError(
            loc,
            "C1032: user defined assignment inside WHERE must be elemental");
      // Non elemental user defined assignment outside of FORALL and WHERE.
      // FIXME: The non elemental user defined assignment case with array
      // arguments must be take into account potential overlap. So far the front
      // end does not add parentheses around the RHS argument in the call as it
      // should according to 15.4.3.4.3 p2.
      Fortran::lower::createSomeExtendedExpression(
          loc, converter, toEvExpr(call), symMap, stmtCtx);
    }
    return {};
  }

  assert(implicitIterSpace.empty() && !explicitIterSpace.isActive() &&
         "subroutine calls are not allowed inside WHERE and FORALL");

  if (isElementalProcWithArrayArgs(call)) {
    ArrayExprLowering::lowerElementalSubroutine(converter, symMap, stmtCtx,
                                                toEvExpr(call));
    return {};
  }
  // Simple subroutine call, with potential alternate return.
  auto res = Fortran::lower::createSomeExtendedExpression(
      loc, converter, toEvExpr(call), symMap, stmtCtx);
  return fir::getBase(res);
}

template <typename A>
fir::ArrayLoadOp genArrayLoad(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              fir::FirOpBuilder &builder, const A *x,
                              Fortran::lower::SymMap &symMap,
                              Fortran::lower::StatementContext &stmtCtx) {
  auto exv = ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(*x);
  mlir::Value addr = fir::getBase(exv);
  mlir::Value shapeOp = builder.createShape(loc, exv);
  mlir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
  return builder.create<fir::ArrayLoadOp>(loc, arrTy, addr, shapeOp,
                                          /*slice=*/mlir::Value{},
                                          fir::getTypeParams(exv));
}
template <>
fir::ArrayLoadOp
genArrayLoad(mlir::Location loc, Fortran::lower::AbstractConverter &converter,
             fir::FirOpBuilder &builder, const Fortran::evaluate::ArrayRef *x,
             Fortran::lower::SymMap &symMap,
             Fortran::lower::StatementContext &stmtCtx) {
  if (x->base().IsSymbol())
    return genArrayLoad(loc, converter, builder, &getLastSym(x->base()), symMap,
                        stmtCtx);
  return genArrayLoad(loc, converter, builder, &x->base().GetComponent(),
                      symMap, stmtCtx);
}

void Fortran::lower::createArrayLoads(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::ExplicitIterSpace &esp, Fortran::lower::SymMap &symMap) {
  std::size_t counter = esp.getCounter();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  Fortran::lower::StatementContext &stmtCtx = esp.stmtContext();
  // Gen the fir.array_load ops.
  auto genLoad = [&](const auto *x) -> fir::ArrayLoadOp {
    return genArrayLoad(loc, converter, builder, x, symMap, stmtCtx);
  };
  if (esp.lhsBases[counter]) {
    auto &base = *esp.lhsBases[counter];
    auto load = std::visit(genLoad, base);
    esp.initialArgs.push_back(load);
    esp.resetInnerArgs();
    esp.bindLoad(base, load);
  }
  for (const auto &base : esp.rhsBases[counter])
    esp.bindLoad(base, std::visit(genLoad, base));
}

void Fortran::lower::createArrayMergeStores(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::ExplicitIterSpace &esp) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  builder.setInsertionPointAfter(esp.getOuterLoop());
  // Gen the fir.array_merge_store ops for all LHS arrays.
  for (auto i : llvm::enumerate(esp.getOuterLoop().getResults()))
    if (std::optional<fir::ArrayLoadOp> ldOpt = esp.getLhsLoad(i.index())) {
      fir::ArrayLoadOp load = *ldOpt;
      builder.create<fir::ArrayMergeStoreOp>(loc, load, i.value(),
                                             load.getMemref(), load.getSlice(),
                                             load.getTypeparams());
    }
  if (esp.loopCleanup) {
    (*esp.loopCleanup)(builder);
    esp.loopCleanup = std::nullopt;
  }
  esp.initialArgs.clear();
  esp.innerArgs.clear();
  esp.outerLoop = std::nullopt;
  esp.resetBindings();
  esp.incrementCounter();
}
