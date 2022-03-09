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
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ComponentPath.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/CustomIntrinsicCall.h"
#include "flang/Lower/DumpEvaluateExpr.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-expr"

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

// The default attempts to balance a modest allocation size with expected user
// input to minimize bounds checks and reallocations during dynamic array
// construction. Some user codes may have very large array constructors for
// which the default can be increased.
static llvm::cl::opt<unsigned> clInitialBufferSize(
    "array-constructor-initial-buffer-size",
    llvm::cl::desc(
        "set the incremental array construction buffer size (default=32)"),
    llvm::cl::init(32u));

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
  if (fir::isDerivedWithLengthParameters(exv))
    TODO(loc, "copy derived type with length parameters");
  mlir::Type type = fir::unwrapPassByRefType(fir::getBase(exv).getType());
  fir::ExtendedValue temp = builder.createTemporary(loc, type);
  fir::factory::genScalarAssignment(builder, loc, temp, exv);
  return temp;
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
      [&](const fir::UnboxedValue &v) -> fir::ExtendedValue {
        if (fir::unwrapRefType(fir::getBase(v).getType())
                .isa<fir::RecordType>())
          return v;
        return builder.create<fir::LoadOp>(loc, fir::getBase(v));
      },
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        TODO(loc, "genLoad for MutableBoxValue");
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        TODO(loc, "genLoad for BoxValue");
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

/// Some auxiliary data for processing initialization in ScalarExprLowering
/// below. This is currently used for generating dense attributed global
/// arrays.
struct InitializerData {
  explicit InitializerData(bool getRawVals = false) : genRawVals{getRawVals} {}
  llvm::SmallVector<mlir::Attribute> rawVals; // initialization raw values
  mlir::Type rawType; // Type of elements processed for rawVals vector.
  bool genRawVals;    // generate the rawVals vector if set.
};

/// If \p arg is the address of a function with a denoted host-association tuple
/// argument, then return the host-associations tuple value of the current
/// procedure. Otherwise, return nullptr.
static mlir::Value
argumentHostAssocs(Fortran::lower::AbstractConverter &converter,
                   mlir::Value arg) {
  if (auto addr = mlir::dyn_cast_or_null<fir::AddrOfOp>(arg.getDefiningOp())) {
    auto &builder = converter.getFirOpBuilder();
    if (auto funcOp = builder.getNamedFunction(addr.getSymbol()))
      if (fir::anyFuncArgsHaveAttr(funcOp, fir::getHostAssocAttrName()))
        return converter.hostAssocTupleValue();
  }
  return {};
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
                              InitializerData *initializer = nullptr)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap},
        inInitializer{initializer} {}

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
              return symMap.lookupSymbol(*sym).toExtendedValue();
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

  /// Generate a real constant with a value `value`.
  template <int KIND>
  mlir::Value genRealConstant(mlir::MLIRContext *context,
                              const llvm::APFloat &value) {
    mlir::Type fltTy = Fortran::lower::convertReal(context, KIND);
    return builder.createRealConstant(getLoc(), fltTy, value);
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

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  ExtValue gen(Fortran::semantics::SymbolRef sym) {
    if (Fortran::lower::SymbolBox val = symMap.lookupSymbol(sym))
      return val.match(
          [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &boxAddr) {
            return fir::factory::genMutableBoxRead(builder, getLoc(), boxAddr);
          },
          [&val](auto &) { return val.toExtendedValue(); });
    LLVM_DEBUG(llvm::dbgs()
               << "unknown symbol: " << sym << "\nmap: " << symMap << '\n');
    fir::emitFatalError(getLoc(), "symbol is not mapped to any IR value");
  }

  ExtValue genLoad(const ExtValue &exv) {
    return ::genLoad(builder, getLoc(), exv);
  }

  ExtValue genval(Fortran::semantics::SymbolRef sym) {
    ExtValue var = gen(sym);
    if (const fir::UnboxedValue *s = var.getUnboxed())
      if (fir::isReferenceLike(s->getType()))
        return genLoad(*s);
    return var;
  }

  ExtValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO(getLoc(), "genval BOZ");
  }

  /// Return indirection to function designated in ProcedureDesignator.
  /// The type of the function indirection is not guaranteed to match the one
  /// of the ProcedureDesignator due to Fortran implicit typing rules.
  ExtValue genval(const Fortran::evaluate::ProcedureDesignator &proc) {
    TODO(getLoc(), "genval ProcedureDesignator");
  }

  ExtValue genval(const Fortran::evaluate::NullPointer &) {
    TODO(getLoc(), "genval NullPointer");
  }

  ExtValue genval(const Fortran::evaluate::StructureConstructor &ctor) {
    TODO(getLoc(), "genval StructureConstructor");
  }

  /// Lowering of an <i>ac-do-variable</i>, which is not a Symbol.
  ExtValue genval(const Fortran::evaluate::ImpliedDoIndex &var) {
    return converter.impliedDoBinding(toStringRef(var.name));
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    ExtValue exv = desc.base().IsSymbol() ? gen(desc.base().GetLastSymbol())
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
      TODO(loc, "Stride inquiry");
    }
    llvm_unreachable("unknown descriptor inquiry");
  }

  ExtValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "genval TypeParamInquiry");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    TODO(getLoc(), "genval ComplexComponent");
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
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    mlir::Type ty = converter.genType(TC, KIND);
    mlir::Value lhs = genunbox(op.left());
    mlir::Value rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    mlir::Value realPartValue = genunbox(op.left());
    return fir::factory::Complex{builder, getLoc()}.createComplex(
        KIND, realPartValue, genunbox(op.right()));
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    TODO(getLoc(), "genval Concat<KIND>");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    TODO(getLoc(), "genval Extremum<TC, KIND>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::SetLength<KIND> &x) {
    TODO(getLoc(), "genval SetLength<KIND>");
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
    TODO(getLoc(), "genval complex comparison");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &op) {
    TODO(getLoc(), "genval char comparison");
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
    mlir::Value operand = genunbox(convert.left());
    return builder.convertWithSemantics(getLoc(), ty, operand);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    TODO(getLoc(), "genval parentheses<A>");
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

  /// Convert a scalar literal constant to IR.
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genScalarLit(
      const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
          &value) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return genIntegerConstant<KIND>(builder.getContext(), value.ToInt64());
    } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
      return genBoolConstant(value.IsTrue());
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      std::string str = value.DumpHexadecimal();
      if constexpr (KIND == 2) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEhalf(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 3) {
        llvm::APFloat floatVal{llvm::APFloatBase::BFloat(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 4) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEsingle(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 10) {
        llvm::APFloat floatVal{llvm::APFloatBase::x87DoubleExtended(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 16) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEquad(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else {
        // convert everything else to double
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEdouble(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      }
    } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
      using TR =
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>;
      Fortran::evaluate::ComplexConstructor<KIND> ctor(
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.REAL()}},
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.AIMAG()}});
      return genunbox(ctor);
    } else /*constexpr*/ {
      llvm_unreachable("unhandled constant");
    }
  }

  /// Generate a raw literal value and store it in the rawVals vector.
  template <Fortran::common::TypeCategory TC, int KIND>
  void
  genRawLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
                &value) {
    mlir::Attribute val;
    assert(inInitializer != nullptr);
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      inInitializer->rawType = converter.genType(TC, KIND);
      val = builder.getIntegerAttr(inInitializer->rawType, value.ToInt64());
    } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
      inInitializer->rawType =
          converter.genType(Fortran::common::TypeCategory::Integer, KIND);
      val = builder.getIntegerAttr(inInitializer->rawType, value.IsTrue());
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      std::string str = value.DumpHexadecimal();
      inInitializer->rawType = converter.genType(TC, KIND);
      llvm::APFloat floatVal{builder.getKindMap().getFloatSemantics(KIND), str};
      val = builder.getFloatAttr(inInitializer->rawType, floatVal);
    } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
      std::string strReal = value.REAL().DumpHexadecimal();
      std::string strImg = value.AIMAG().DumpHexadecimal();
      inInitializer->rawType = converter.genType(TC, KIND);
      llvm::APFloat realVal{builder.getKindMap().getFloatSemantics(KIND),
                            strReal};
      val = builder.getFloatAttr(inInitializer->rawType, realVal);
      inInitializer->rawVals.push_back(val);
      llvm::APFloat imgVal{builder.getKindMap().getFloatSemantics(KIND),
                           strImg};
      val = builder.getFloatAttr(inInitializer->rawType, imgVal);
    }
    inInitializer->rawVals.push_back(val);
  }

  /// Convert a ascii scalar literal CHARACTER to IR. (specialization)
  ExtValue
  genAsciiScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                        Fortran::common::TypeCategory::Character, 1>> &value,
                    int64_t len) {
    assert(value.size() == static_cast<std::uint64_t>(len));
    // Outline character constant in ro data if it is not in an initializer.
    if (!inInitializer)
      return fir::factory::createStringLiteral(builder, getLoc(), value);
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    fir::StringLitOp stringLit = builder.createStringLitOp(getLoc(), value);
    mlir::Value lenp = builder.createIntegerConstant(
        getLoc(), builder.getCharacterLengthType(), len);
    return fir::CharBoxValue{stringLit.getResult(), lenp};
  }
  /// Convert a non ascii scalar literal CHARACTER to IR. (specialization)
  template <int KIND>
  ExtValue
  genScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                   Fortran::common::TypeCategory::Character, KIND>> &value,
               int64_t len) {
    using ET = typename std::decay_t<decltype(value)>::value_type;
    if constexpr (KIND == 1) {
      return genAsciiScalarLit(value, len);
    }
    fir::CharacterType type =
        fir::CharacterType::get(builder.getContext(), KIND, len);
    auto consLit = [&]() -> fir::StringLitOp {
      mlir::MLIRContext *context = builder.getContext();
      std::int64_t size = static_cast<std::int64_t>(value.size());
      mlir::ShapedType shape = mlir::VectorType::get(
          llvm::ArrayRef<std::int64_t>{size},
          mlir::IntegerType::get(builder.getContext(), sizeof(ET) * 8));
      auto strAttr = mlir::DenseElementsAttr::get(
          shape, llvm::ArrayRef<ET>{value.data(), value.size()});
      auto valTag = mlir::StringAttr::get(context, fir::StringLitOp::value());
      mlir::NamedAttribute dataAttr(valTag, strAttr);
      auto sizeTag = mlir::StringAttr::get(context, fir::StringLitOp::size());
      mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
      llvm::SmallVector<mlir::NamedAttribute> attrs = {dataAttr, sizeAttr};
      return builder.create<fir::StringLitOp>(
          getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
    };

    mlir::Value lenp = builder.createIntegerConstant(
        getLoc(), builder.getCharacterLengthType(), len);
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    if (inInitializer)
      return fir::CharBoxValue{consLit().getResult(), lenp};

    // Otherwise, the string is in a plain old expression so "outline" the value
    // by hashconsing it to a constant literal object.

    // FIXME: For wider char types, lowering ought to use an array of i16 or
    // i32. But for now, lowering just fakes that the string value is a range of
    // i8 to get it past the C++ compiler.
    std::string globalName =
        fir::factory::uniqueCGIdent("cl", (const char *)value.c_str());
    fir::GlobalOp global = builder.getNamedGlobal(globalName);
    if (!global)
      global = builder.createGlobalConstant(
          getLoc(), type, globalName,
          [&](fir::FirOpBuilder &builder) {
            fir::StringLitOp str = consLit();
            builder.create<fir::HasValueOp>(getLoc(), str);
          },
          builder.createLinkOnceLinkage());
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    return fir::CharBoxValue{addr, lenp};
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genArrayLit(
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &con) {
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    Fortran::evaluate::ConstantSubscript size =
        Fortran::evaluate::GetSize(con.shape());
    fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
    mlir::Type eleTy;
    if constexpr (TC == Fortran::common::TypeCategory::Character)
      eleTy = converter.genType(TC, KIND, {con.LEN()});
    else
      eleTy = converter.genType(TC, KIND);
    auto arrayTy = fir::SequenceType::get(shape, eleTy);
    mlir::Value array;
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    if (!inInitializer || !inInitializer->genRawVals) {
      array = builder.create<fir::UndefOp>(loc, arrayTy);
      for (auto [lb, extent] : llvm::zip(con.lbounds(), shape)) {
        lbounds.push_back(builder.createIntegerConstant(loc, idxTy, lb - 1));
        extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
      }
    }
    if (size == 0) {
      if constexpr (TC == Fortran::common::TypeCategory::Character) {
        mlir::Value len = builder.createIntegerConstant(loc, idxTy, con.LEN());
        return fir::CharArrayBoxValue{array, len, extents, lbounds};
      } else {
        return fir::ArrayBoxValue{array, extents, lbounds};
      }
    }
    Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
    auto createIdx = [&]() {
      llvm::SmallVector<mlir::Attribute> idx;
      for (size_t i = 0; i < subscripts.size(); ++i)
        idx.push_back(
            builder.getIntegerAttr(idxTy, subscripts[i] - con.lbounds()[i]));
      return idx;
    };
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      assert(array && "array must not be nullptr");
      do {
        mlir::Value elementVal =
            fir::getBase(genScalarLit<KIND>(con.At(subscripts), con.LEN()));
        array = builder.create<fir::InsertValueOp>(
            loc, arrayTy, array, elementVal, builder.getArrayAttr(createIdx()));
      } while (con.IncrementSubscripts(subscripts));
      mlir::Value len = builder.createIntegerConstant(loc, idxTy, con.LEN());
      return fir::CharArrayBoxValue{array, len, extents, lbounds};
    } else {
      llvm::SmallVector<mlir::Attribute> rangeStartIdx;
      uint64_t rangeSize = 0;
      do {
        if (inInitializer && inInitializer->genRawVals) {
          genRawLit<TC, KIND>(con.At(subscripts));
          continue;
        }
        auto getElementVal = [&]() {
          return builder.createConvert(
              loc, eleTy,
              fir::getBase(genScalarLit<TC, KIND>(con.At(subscripts))));
        };
        Fortran::evaluate::ConstantSubscripts nextSubscripts = subscripts;
        bool nextIsSame = con.IncrementSubscripts(nextSubscripts) &&
                          con.At(subscripts) == con.At(nextSubscripts);
        if (!rangeSize && !nextIsSame) { // single (non-range) value
          array = builder.create<fir::InsertValueOp>(
              loc, arrayTy, array, getElementVal(),
              builder.getArrayAttr(createIdx()));
        } else if (!rangeSize) { // start a range
          rangeStartIdx = createIdx();
          rangeSize = 1;
        } else if (nextIsSame) { // expand a range
          ++rangeSize;
        } else { // end a range
          llvm::SmallVector<int64_t> rangeBounds;
          llvm::SmallVector<mlir::Attribute> idx = createIdx();
          for (size_t i = 0; i < idx.size(); ++i) {
            rangeBounds.push_back(rangeStartIdx[i]
                                      .cast<mlir::IntegerAttr>()
                                      .getValue()
                                      .getSExtValue());
            rangeBounds.push_back(
                idx[i].cast<mlir::IntegerAttr>().getValue().getSExtValue());
          }
          array = builder.create<fir::InsertOnRangeOp>(
              loc, arrayTy, array, getElementVal(),
              builder.getIndexVectorAttr(rangeBounds));
          rangeSize = 0;
        }
      } while (con.IncrementSubscripts(subscripts));
      return fir::ArrayBoxValue{array, extents, lbounds};
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    if (con.Rank() > 0)
      return genArrayLit(con);
    std::optional<Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>>
        opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      return genScalarLit<KIND>(opt.value(), con.LEN());
    } else {
      return genScalarLit<TC, KIND>(opt.value());
    }
  }

  fir::ExtendedValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    TODO(getLoc(), "genval constant derived");
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    TODO(getLoc(), "genval ArrayConstructor<A>");
  }

  ExtValue gen(const Fortran::evaluate::ComplexPart &x) {
    TODO(getLoc(), "gen ComplexPart");
  }
  ExtValue genval(const Fortran::evaluate::ComplexPart &x) {
    TODO(getLoc(), "genval ComplexPart");
  }

  ExtValue gen(const Fortran::evaluate::Substring &s) {
    TODO(getLoc(), "gen Substring");
  }
  ExtValue genval(const Fortran::evaluate::Substring &ss) {
    TODO(getLoc(), "genval Substring");
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
    TODO(getLoc(), "gen DataRef");
  }
  ExtValue genval(const Fortran::evaluate::DataRef &dref) {
    TODO(getLoc(), "genval DataRef");
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
    if (!cmpt.GetLastSymbol().test(
            Fortran::semantics::Symbol::Flag::ParentComp))
      list.push_front(&cmpt);
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Component &x) {
              if (Fortran::semantics::IsAllocatableOrPointer(x.GetLastSymbol()))
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
      const Fortran::semantics::Symbol &sym = field->GetLastSymbol();
      llvm::StringRef name = toStringRef(sym.name());
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(
          loc, fldTy, name, recTy, fir::getTypeParams(obj)));
      ty = recTy.getType(name);
    }
    ty = builder.getRefType(ty);
    return fir::factory::componentToExtendedValue(
        builder, loc,
        builder.create<fir::CoordinateOp>(loc, ty, fir::getBase(obj),
                                          coorArgs));
  }

  ExtValue gen(const Fortran::evaluate::Component &cmpt) {
    TODO(getLoc(), "gen Component");
  }
  ExtValue genval(const Fortran::evaluate::Component &cmpt) {
    TODO(getLoc(), "genval Component");
  }

  ExtValue genval(const Fortran::semantics::Bound &bound) {
    TODO(getLoc(), "genval Bound");
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
    auto seqTy =
        fir::dyn_cast_ptrOrBoxEleTy(base.getType()).cast<fir::SequenceType>();
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
          fir::emitFatalError(loc, "internal: array lowering failed");
        });
  }

  ExtValue gen(const Fortran::evaluate::ArrayRef &aref) {
    ExtValue base = aref.base().IsSymbol() ? gen(aref.base().GetFirstSymbol())
                                           : gen(aref.base().GetComponent());
    return genCoordinateOp(base, aref);
  }
  ExtValue genval(const Fortran::evaluate::ArrayRef &aref) {
    return genLoad(gen(aref));
  }

  ExtValue gen(const Fortran::evaluate::CoarrayRef &coref) {
    TODO(getLoc(), "gen CoarrayRef");
  }
  ExtValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    TODO(getLoc(), "genval CoarrayRef");
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
    TODO(getLoc(), "genType Derived Type");
  }

  /// Lower a function reference
  template <typename A>
  ExtValue genFunctionRef(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    if (!funcRef.GetType().has_value())
      fir::emitFatalError(getLoc(), "internal: a function must have a type");
    mlir::Type resTy = genType(*funcRef.GetType());
    return genProcedureRef(funcRef, {resTy});
  }

  /// Lower function call `funcRef` and return a reference to the resultant
  /// value. This is required for lowering expressions such as `f1(f2(v))`.
  template <typename A>
  ExtValue gen(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    ExtValue retVal = genFunctionRef(funcRef);
    mlir::Value retValBase = fir::getBase(retVal);
    if (fir::conformsWithPassByRef(retValBase.getType()))
      return retVal;
    auto mem = builder.create<fir::AllocaOp>(getLoc(), retValBase.getType());
    builder.create<fir::StoreOp>(getLoc(), retValBase, mem);
    return fir::substBase(retVal, mem.getResult());
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

  /// Helper to package a Value and its properties into an ExtendedValue.
  static ExtValue toExtendedValue(mlir::Location loc, mlir::Value base,
                                  llvm::ArrayRef<mlir::Value> extents,
                                  llvm::ArrayRef<mlir::Value> lengths) {
    mlir::Type type = base.getType();
    if (type.isa<fir::BoxType>())
      return fir::BoxValue(base, /*lbounds=*/{}, lengths, extents);
    type = fir::unwrapRefType(type);
    if (type.isa<fir::BoxType>())
      return fir::MutableBoxValue(base, lengths, /*mutableProperties*/ {});
    if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
      if (seqTy.getDimension() != extents.size())
        fir::emitFatalError(loc, "incorrect number of extents for array");
      if (seqTy.getEleTy().isa<fir::CharacterType>()) {
        if (lengths.empty())
          fir::emitFatalError(loc, "missing length for character");
        assert(lengths.size() == 1);
        return fir::CharArrayBoxValue(base, lengths[0], extents);
      }
      return fir::ArrayBoxValue(base, extents);
    }
    if (type.isa<fir::CharacterType>()) {
      if (lengths.empty())
        fir::emitFatalError(loc, "missing length for character");
      assert(lengths.size() == 1);
      return fir::CharBoxValue(base, lengths[0]);
    }
    return base;
  }

  // Find the argument that corresponds to the host associations.
  // Verify some assumptions about how the signature was built here.
  [[maybe_unused]] static unsigned findHostAssocTuplePos(mlir::FuncOp fn) {
    // Scan the argument list from last to first as the host associations are
    // appended for now.
    for (unsigned i = fn.getNumArguments(); i > 0; --i)
      if (fn.getArgAttr(i - 1, fir::getHostAssocAttrName())) {
        // Host assoc tuple must be last argument (for now).
        assert(i == fn.getNumArguments() && "tuple must be last");
        return i - 1;
      }
    llvm_unreachable("anyFuncArgsHaveAttr failed");
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
        fir::factory::getExtents(builder, loc, mold);
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
      TODO(loc, "Creating temporary for derived type with length parameters");
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
                           llvm::Optional<mlir::Type> resultType) {
    ExtValue res = genRawProcedureRef(procRef, resultType);
    return res;
  }

  /// Given a call site for which the arguments were already lowered, generate
  /// the call and return the result. This function deals with explicit result
  /// allocation and lowering if needed. It also deals with passing the host
  /// link to internal procedures.
  ExtValue genCallOpAndResult(Fortran::lower::CallerInterface &caller,
                              mlir::FunctionType callSiteType,
                              llvm::Optional<mlir::Type> resultType) {
    mlir::Location loc = getLoc();
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
    // Handle cases where caller must allocate the result or a fir.box for it.
    bool mustPopSymMap = false;
    if (caller.mustMapInterfaceSymbols()) {
      symMap.pushScope();
      mustPopSymMap = true;
      Fortran::lower::mapCallInterfaceSymbols(converter, caller, symMap);
    }
    // If this is an indirect call, retrieve the function address. Also retrieve
    // the result length if this is a character function (note that this length
    // will be used only if there is no explicit length in the local interface).
    mlir::Value funcPointer;
    mlir::Value charFuncPointerLength;
    if (const Fortran::semantics::Symbol *sym =
            caller.getIfIndirectCallSymbol()) {
      funcPointer = symMap.lookupSymbol(*sym).getAddr();
      if (!funcPointer)
        fir::emitFatalError(loc, "failed to find indirect call symbol address");
      if (fir::isCharacterProcedureTuple(funcPointer.getType(),
                                         /*acceptRawFunc=*/false))
        std::tie(funcPointer, charFuncPointerLength) =
            fir::factory::extractCharacterProcedureTuple(builder, loc,
                                                         funcPointer);
    }

    mlir::IndexType idxTy = builder.getIndexType();
    auto lowerSpecExpr = [&](const auto &expr) -> mlir::Value {
      return builder.createConvert(
          loc, idxTy, fir::getBase(converter.genExprValue(expr, stmtCtx)));
    };
    llvm::SmallVector<mlir::Value> resultLengths;
    auto allocatedResult = [&]() -> llvm::Optional<ExtValue> {
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> lengths;
      if (!caller.callerAllocateResult())
        return {};
      mlir::Type type = caller.getResultStorageType();
      if (type.isa<fir::SequenceType>())
        caller.walkResultExtents([&](const Fortran::lower::SomeExpr &e) {
          extents.emplace_back(lowerSpecExpr(e));
        });
      caller.walkResultLengths([&](const Fortran::lower::SomeExpr &e) {
        lengths.emplace_back(lowerSpecExpr(e));
      });

      // Result length parameters should not be provided to box storage
      // allocation and save_results, but they are still useful information to
      // keep in the ExtendedValue if non-deferred.
      if (!type.isa<fir::BoxType>()) {
        if (fir::isa_char(fir::unwrapSequenceType(type)) && lengths.empty()) {
          // Calling an assumed length function. This is only possible if this
          // is a call to a character dummy procedure.
          if (!charFuncPointerLength)
            fir::emitFatalError(loc, "failed to retrieve character function "
                                     "length while calling it");
          lengths.push_back(charFuncPointerLength);
        }
        resultLengths = lengths;
      }

      if (!extents.empty() || !lengths.empty()) {
        auto *bldr = &converter.getFirOpBuilder();
        auto stackSaveFn = fir::factory::getLlvmStackSave(builder);
        auto stackSaveSymbol = bldr->getSymbolRefAttr(stackSaveFn.getName());
        mlir::Value sp =
            bldr->create<fir::CallOp>(loc, stackSaveFn.getType().getResults(),
                                      stackSaveSymbol, mlir::ValueRange{})
                .getResult(0);
        stmtCtx.attachCleanup([bldr, loc, sp]() {
          auto stackRestoreFn = fir::factory::getLlvmStackRestore(*bldr);
          auto stackRestoreSymbol =
              bldr->getSymbolRefAttr(stackRestoreFn.getName());
          bldr->create<fir::CallOp>(loc, stackRestoreFn.getType().getResults(),
                                    stackRestoreSymbol, mlir::ValueRange{sp});
        });
      }
      mlir::Value temp =
          builder.createTemporary(loc, type, ".result", extents, resultLengths);
      return toExtendedValue(loc, temp, extents, lengths);
    }();

    if (mustPopSymMap)
      symMap.popScope();

    // Place allocated result or prepare the fir.save_result arguments.
    mlir::Value arrayResultShape;
    if (allocatedResult) {
      if (std::optional<Fortran::lower::CallInterface<
              Fortran::lower::CallerInterface>::PassedEntity>
              resultArg = caller.getPassedResult()) {
        if (resultArg->passBy == PassBy::AddressAndLength)
          caller.placeAddressAndLengthInput(*resultArg,
                                            fir::getBase(*allocatedResult),
                                            fir::getLen(*allocatedResult));
        else if (resultArg->passBy == PassBy::BaseAddress)
          caller.placeInput(*resultArg, fir::getBase(*allocatedResult));
        else
          fir::emitFatalError(
              loc, "only expect character scalar result to be passed by ref");
      } else {
        assert(caller.mustSaveResult());
        arrayResultShape = allocatedResult->match(
            [&](const fir::CharArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const fir::ArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const auto &) { return mlir::Value{}; });
      }
    }

    // In older Fortran, procedure argument types are inferred. This may lead
    // different view of what the function signature is in different locations.
    // Casts are inserted as needed below to accommodate this.

    // The mlir::FuncOp type prevails, unless it has a different number of
    // arguments which can happen in legal program if it was passed as a dummy
    // procedure argument earlier with no further type information.
    mlir::SymbolRefAttr funcSymbolAttr;
    bool addHostAssociations = false;
    if (!funcPointer) {
      mlir::FunctionType funcOpType = caller.getFuncOp().getType();
      mlir::SymbolRefAttr symbolAttr =
          builder.getSymbolRefAttr(caller.getMangledName());
      if (callSiteType.getNumResults() == funcOpType.getNumResults() &&
          callSiteType.getNumInputs() + 1 == funcOpType.getNumInputs() &&
          fir::anyFuncArgsHaveAttr(caller.getFuncOp(),
                                   fir::getHostAssocAttrName())) {
        // The number of arguments is off by one, and we're lowering a function
        // with host associations. Modify call to include host associations
        // argument by appending the value at the end of the operands.
        assert(funcOpType.getInput(findHostAssocTuplePos(caller.getFuncOp())) ==
               converter.hostAssocTupleValue().getType());
        addHostAssociations = true;
      }
      if (!addHostAssociations &&
          (callSiteType.getNumResults() != funcOpType.getNumResults() ||
           callSiteType.getNumInputs() != funcOpType.getNumInputs())) {
        // Deal with argument number mismatch by making a function pointer so
        // that function type cast can be inserted. Do not emit a warning here
        // because this can happen in legal program if the function is not
        // defined here and it was first passed as an argument without any more
        // information.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else if (callSiteType.getResults() != funcOpType.getResults()) {
        // Implicit interface result type mismatch are not standard Fortran, but
        // some compilers are not complaining about it.  The front end is not
        // protecting lowering from this currently. Support this with a
        // discouraging warning.
        LLVM_DEBUG(mlir::emitWarning(
            loc, "a return type mismatch is not standard compliant and may "
                 "lead to undefined behavior."));
        // Cast the actual function to the current caller implicit type because
        // that is the behavior we would get if we could not see the definition.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else {
        funcSymbolAttr = symbolAttr;
      }
    }

    mlir::FunctionType funcType =
        funcPointer ? callSiteType : caller.getFuncOp().getType();
    llvm::SmallVector<mlir::Value> operands;
    // First operand of indirect call is the function pointer. Cast it to
    // required function type for the call to handle procedures that have a
    // compatible interface in Fortran, but that have different signatures in
    // FIR.
    if (funcPointer) {
      operands.push_back(
          funcPointer.getType().isa<fir::BoxProcType>()
              ? builder.create<fir::BoxAddrOp>(loc, funcType, funcPointer)
              : builder.createConvert(loc, funcType, funcPointer));
    }

    // Deal with potential mismatches in arguments types. Passing an array to a
    // scalar argument should for instance be tolerated here.
    bool callingImplicitInterface = caller.canBeCalledViaImplicitInterface();
    for (auto [fst, snd] :
         llvm::zip(caller.getInputs(), funcType.getInputs())) {
      // When passing arguments to a procedure that can be called an implicit
      // interface, allow character actual arguments to be passed to dummy
      // arguments of any type and vice versa
      mlir::Value cast;
      auto *context = builder.getContext();
      if (snd.isa<fir::BoxProcType>() &&
          fst.getType().isa<mlir::FunctionType>()) {
        auto funcTy = mlir::FunctionType::get(context, llvm::None, llvm::None);
        auto boxProcTy = builder.getBoxProcType(funcTy);
        if (mlir::Value host = argumentHostAssocs(converter, fst)) {
          cast = builder.create<fir::EmboxProcOp>(
              loc, boxProcTy, llvm::ArrayRef<mlir::Value>{fst, host});
        } else {
          cast = builder.create<fir::EmboxProcOp>(loc, boxProcTy, fst);
        }
      } else {
        cast = builder.convertWithSemantics(loc, snd, fst,
                                            callingImplicitInterface);
      }
      operands.push_back(cast);
    }

    // Add host associations as necessary.
    if (addHostAssociations)
      operands.push_back(converter.hostAssocTupleValue());

    auto call = builder.create<fir::CallOp>(loc, funcType.getResults(),
                                            funcSymbolAttr, operands);

    if (caller.mustSaveResult())
      builder.create<fir::SaveResultOp>(
          loc, call.getResult(0), fir::getBase(allocatedResult.getValue()),
          arrayResultShape, resultLengths);

    if (allocatedResult) {
      allocatedResult->match(
          [&](const fir::MutableBoxValue &box) {
            if (box.isAllocatable()) {
              // 9.7.3.2 point 4. Finalize allocatables.
              fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
              stmtCtx.attachCleanup([bldr, loc, box]() {
                fir::factory::genFinalization(*bldr, loc, box);
              });
            }
          },
          [](const auto &) {});
      return *allocatedResult;
    }

    if (!resultType.hasValue())
      return mlir::Value{}; // subroutine call
    // For now, Fortran return values are implemented with a single MLIR
    // function return value.
    assert(call.getNumResults() == 1 &&
           "Expected exactly one result in FUNCTION call");
    return call.getResult(0);
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
    mlir::Location loc = getLoc();
    if (expr.Rank() > 0)
      TODO(loc, "genTempExtAddr array");
    return genExtValue(expr).match(
        [&](const fir::CharBoxValue &boxChar) -> ExtValue {
          TODO(loc, "genTempExtAddr CharBoxValue");
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
          assert(!x.hasRank() && x.isDerived());
          if (x.isDerivedWithLengthParameters())
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
    llvm::Optional<mlir::Value> restrictCopyAndFreeAtRuntime;
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
                     llvm::Optional<mlir::Value> restrictCopyAtRuntime) {
    if (!restrictCopyAtRuntime) {
      ExtValue temp = genArrayTempFromMold(actualArg, ".copyinout");
      if (arg.mayBeReadByCall())
        genArrayCopy(temp, actualArg);
      copyOutPairs.emplace_back(CopyOutPair{
          actualArg, temp, arg.mayBeModifiedByCall(), restrictCopyAtRuntime});
      return temp;
    }
    // Otherwise, need to be careful to only copy-in if allowed at runtime.
    mlir::Location loc = getLoc();
    auto addrType = fir::HeapType::get(
        fir::unwrapPassByRefType(fir::getBase(actualArg).getType()));
    mlir::Value addr =
        builder
            .genIfOp(loc, {addrType}, *restrictCopyAtRuntime,
                     /*withElseRegion=*/true)
            .genThen([&]() {
              auto temp = genArrayTempFromMold(actualArg, ".copyinout");
              if (arg.mayBeReadByCall())
                genArrayCopy(temp, actualArg);
              builder.create<fir::ResultOp>(loc, fir::getBase(temp));
            })
            .genElse([&]() {
              auto nullPtr = builder.createNullConstant(loc, addrType);
              builder.create<fir::ResultOp>(loc, nullPtr);
            })
            .getResults()[0];
    // Associate the temp address with actualArg lengths and extents.
    fir::ExtendedValue temp = fir::substBase(readIfBoxValue(actualArg), addr);
    copyOutPairs.emplace_back(CopyOutPair{
        actualArg, temp, arg.mayBeModifiedByCall(), restrictCopyAtRuntime});
    return temp;
  }

  /// Lower a non-elemental procedure reference.
  ExtValue genRawProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                              llvm::Optional<mlir::Type> resultType) {
    mlir::Location loc = getLoc();
    if (isElementalProcWithArrayArgs(procRef))
      fir::emitFatalError(loc, "trying to lower elemental procedure with array "
                               "arguments as normal procedure");
    if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
            procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType);

    if (isStatementFunctionCall(procRef))
      TODO(loc, "Lower statement function call");

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
        caller.placeInput(arg, builder.create<fir::AbsentOp>(loc, argTy));
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

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
          assert(boxTy && boxTy.isa<fir::BoxType>() &&
                 "must be a fir.box type");
          mlir::Value boxStorage = builder.createTemporary(loc, boxTy);
          mlir::Value nullBox = fir::factory::createUnallocatedBox(
              builder, loc, boxTy, /*nonDeferredParams=*/{});
          builder.create<fir::StoreOp>(loc, nullBox, boxStorage);
          caller.placeInput(arg, boxStorage);
          continue;
        }
        fir::MutableBoxValue mutableBox = genMutableBoxValue(*expr);
        mlir::Value irBox =
            fir::factory::getMutableIRBox(builder, loc, mutableBox);
        caller.placeInput(arg, irBox);
        if (arg.mayBeModifiedByCall())
          mutableModifiedByCall.emplace_back(std::move(mutableBox));
        continue;
      }
      const bool actualArgIsVariable = Fortran::evaluate::IsVariable(*expr);
      if (arg.passBy == PassBy::BaseAddress || arg.passBy == PassBy::BoxChar) {
        const bool actualIsSimplyContiguous =
            !actualArgIsVariable || Fortran::evaluate::IsSimplyContiguous(
                                        *expr, converter.getFoldingContext());
        auto argAddr = [&]() -> ExtValue {
          ExtValue baseAddr;
          if (actualArgIsVariable && arg.isOptional()) {
            if (Fortran::evaluate::IsAllocatableOrPointerObject(
                    *expr, converter.getFoldingContext())) {
              TODO(loc, "Allocatable or pointer argument");
            }
            if (const Fortran::semantics::Symbol *wholeSymbol =
                    Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(
                        *expr))
              if (Fortran::semantics::IsOptional(*wholeSymbol)) {
                TODO(loc, "procedureref optional arg");
              }
            // Fall through: The actual argument can safely be
            // copied-in/copied-out without any care if needed.
          }
          if (actualArgIsVariable && expr->Rank() > 0) {
            ExtValue box = genBoxArg(*expr);
            if (!actualIsSimplyContiguous)
              return genCopyIn(box, arg, copyOutPairs,
                               /*restrictCopyAtRuntime=*/llvm::None);
            // Contiguous: just use the box we created above!
            // This gets "unboxed" below, if needed.
            return box;
          }
          // Actual argument is a non optional/non pointer/non allocatable
          // scalar.
          if (actualArgIsVariable)
            return genExtAddr(*expr);
          // Actual argument is not a variable. Make sure a variable address is
          // not passed.
          return genTempExtAddr(*expr);
        }();
        // Scalar and contiguous expressions may be lowered to a fir.box,
        // either to account for potential polymorphism, or because lowering
        // did not account for some contiguity hints.
        // Here, polymorphism does not matter (an entity of the declared type
        // is passed, not one of the dynamic type), and the expr is known to
        // be simply contiguous, so it is safe to unbox it and pass the
        // address without making a copy.
        argAddr = readIfBoxValue(argAddr);

        if (arg.passBy == PassBy::BaseAddress) {
          caller.placeInput(arg, fir::getBase(argAddr));
        } else {
          assert(arg.passBy == PassBy::BoxChar);
          auto helper = fir::factory::CharacterExprHelper{builder, loc};
          auto boxChar = argAddr.match(
              [&](const fir::CharBoxValue &x) { return helper.createEmbox(x); },
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
        // Before lowering to an address, handle the allocatable/pointer actual
        // argument to optional fir.box dummy. It is legal to pass
        // unallocated/disassociated entity to an optional. In this case, an
        // absent fir.box must be created instead of a fir.box with a null value
        // (Fortran 2018 15.5.2.12 point 1).
        if (arg.isOptional() && Fortran::evaluate::IsAllocatableOrPointerObject(
                                    *expr, converter.getFoldingContext())) {
          TODO(loc, "optional allocatable or pointer argument");
        } else {
          // Make sure a variable address is only passed if the expression is
          // actually a variable.
          mlir::Value box =
              actualArgIsVariable
                  ? builder.createBox(loc, genBoxArg(*expr))
                  : builder.createBox(getLoc(), genTempExtAddr(*expr));
          caller.placeInput(arg, box);
        }
      } else if (arg.passBy == PassBy::AddressAndLength) {
        ExtValue argRef = genExtAddr(*expr);
        caller.placeAddressAndLengthInput(arg, fir::getBase(argRef),
                                          fir::getLen(argRef));
      } else if (arg.passBy == PassBy::CharProcTuple) {
        TODO(loc, "procedureref CharProcTuple");
      } else {
        TODO(loc, "pass by value in non elemental function call");
      }
    }

    ExtValue result = genCallOpAndResult(caller, callSiteType, resultType);

    // // Copy-out temps that were created for non contiguous variable arguments
    // if
    // // needed.
    // for (const auto &copyOutPair : copyOutPairs)
    //   genCopyOut(copyOutPair);

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
    llvm::Optional<mlir::Type> resTy;
    if (procRef.hasAlternateReturns())
      resTy = builder.getIndexType();
    return genProcedureRef(procRef, resTy);
  }

  /// Helper to lower intrinsic arguments for inquiry intrinsic.
  ExtValue
  lowerIntrinsicArgumentAsInquired(const Fortran::lower::SomeExpr &expr) {
    if (Fortran::evaluate::IsAllocatableOrPointerObject(
            expr, converter.getFoldingContext()))
      return genMutableBoxValue(expr);
    return gen(expr);
  }

  /// Helper to lower intrinsic arguments to a fir::BoxValue.
  /// It preserves all the non default lower bounds/non deferred length
  /// parameter information.
  ExtValue lowerIntrinsicArgumentAsBox(const Fortran::lower::SomeExpr &expr) {
    mlir::Location loc = getLoc();
    ExtValue exv = genBoxArg(expr);
    mlir::Value box = builder.createBox(loc, exv);
    return fir::BoxValue(
        box, fir::factory::getNonDefaultLowerBounds(builder, loc, exv),
        fir::factory::getNonDeferredLengthParams(exv));
  }

  /// Generate a call to an intrinsic function.
  ExtValue
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  llvm::Optional<mlir::Type> resultType) {
    llvm::SmallVector<ExtValue> operands;

    llvm::StringRef name = intrinsic.name;
    mlir::Location loc = getLoc();

    const Fortran::lower::IntrinsicArgumentLoweringRules *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back(Fortran::lower::getAbsentIntrinsicArgument());
        continue;
      }
      if (!argLowering) {
        // No argument lowering instruction, lower by value.
        operands.emplace_back(genval(*expr));
        continue;
      }
      // Ad-hoc argument lowering handling.
      Fortran::lower::ArgLoweringRule argRules =
          Fortran::lower::lowerIntrinsicArgumentAs(loc, *argLowering,
                                                   dummy.name);
      if (argRules.handleDynamicOptional &&
          Fortran::evaluate::MayBePassedAsAbsentOptional(
              *expr, converter.getFoldingContext())) {
        ExtValue optional = lowerIntrinsicArgumentAsInquired(*expr);
        mlir::Value isPresent = genActualIsPresentTest(builder, loc, optional);
        switch (argRules.lowerAs) {
        case Fortran::lower::LowerIntrinsicArgAs::Value:
          operands.emplace_back(
              genOptionalValue(builder, loc, optional, isPresent));
          continue;
        case Fortran::lower::LowerIntrinsicArgAs::Addr:
          operands.emplace_back(
              genOptionalAddr(builder, loc, optional, isPresent));
          continue;
        case Fortran::lower::LowerIntrinsicArgAs::Box:
          operands.emplace_back(
              genOptionalBox(builder, loc, optional, isPresent));
          continue;
        case Fortran::lower::LowerIntrinsicArgAs::Inquired:
          operands.emplace_back(optional);
          continue;
        }
        llvm_unreachable("bad switch");
      }
      switch (argRules.lowerAs) {
      case Fortran::lower::LowerIntrinsicArgAs::Value:
        operands.emplace_back(genval(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Addr:
        operands.emplace_back(gen(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Box:
        operands.emplace_back(lowerIntrinsicArgumentAsBox(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Inquired:
        operands.emplace_back(lowerIntrinsicArgumentAsInquired(*expr));
        continue;
      }
      llvm_unreachable("bad switch");
    }
    // Let the intrinsic library lower the intrinsic procedure call
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name, resultType,
                                            operands, stmtCtx);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x) ||
        inInitializer)
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    return asArray(x);
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
        isTransformationalRef(x))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
    if (useBoxArg)
      return asArrayArg(x);
    return asArray(x);
  }

  template <typename A>
  bool isScalar(const A &x) {
    return x.Rank() == 0;
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
  InitializerData *inInitializer = nullptr;
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
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut};
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
    fir::MutableBoxValue mutableBox =
        createMutableBox(loc, converter, lhs, symMap);
    mlir::Type resultTy = converter.genType(rhs);
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
        mutableBox.isDerivedWithLengthParameters())
      TODO(loc, "gather rhs length parameters in assignment to allocatable");

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
      std::vector<mlir::Value> lbs =
          fir::factory::getOrigins(arrayOperands[0].shape);
      lbounds.append(lbs.begin(), lbs.end());
    }
    fir::factory::MutableBoxReallocation realloc =
        fir::factory::genReallocIfNeeded(builder, loc, mutableBox, destShape,
                                         lengthParams);
    // Create ArrayLoad for the mutable box and save it into `destination`.
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    ccStoreToDest = genarr(realloc.newValue);
    // If the rhs is scalar, get shape from the allocatable ArrayLoad.
    if (destShape.empty())
      destShape = getShape(destination);
    // Finish lowering the loop nest.
    assert(destination && "destination must have been set");
    ExtValue exv = lowerArrayExpression(rhsCC, resultTy);
    if (explicitSpaceIsActive()) {
      explicitSpace->finalizeContext();
      builder.create<fir::ResultOp>(loc, fir::getBase(exv));
    } else {
      builder.create<fir::ArrayMergeStoreOp>(
          loc, destination, fir::getBase(exv), destination.getMemref(),
          destination.getSlice(), destination.getTypeparams());
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
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          ExtValue exv = f(IterationSpace{});
          if (fir::getBase(exv).getType().template isa<fir::BoxType>())
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

  /// Build the iteration space into which the array expression will be
  /// lowered. The resultType is used to create a temporary, if needed.
  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint>
  genIterSpace(mlir::Type resultType) {
    mlir::Location loc = getLoc();
    llvm::SmallVector<mlir::Value> shape = genIterationShape();
    if (!destination) {
      // Allocate storage for the result if it is not already provided.
      destination = createAndLoadSomeArrayTemp(resultType, shape);
    }

    // Generate the lazy mask allocation, if one was given.
    if (ccPrelude.hasValue())
      ccPrelude.getValue()(shape);

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
    if (ccLoadDest.hasValue())
      return ccLoadDest.getValue()(shape);
    auto seqTy = type.dyn_cast<fir::SequenceType>();
    assert(seqTy && "must be an array");
    mlir::Location loc = getLoc();
    // TODO: Need to thread the length parameters here. For character, they may
    // differ from the operands length (e.g concatenation). So the array loads
    // type parameters are not enough.
    if (auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>())
      if (charTy.hasDynamicLen())
        TODO(loc, "character array expression temp with dynamic length");
    if (auto recTy = seqTy.getEleTy().dyn_cast<fir::RecordType>())
      if (recTy.getNumLenParams() > 0)
        TODO(loc, "derived type array expression temp with length parameters");
    mlir::Value temp = seqTy.hasConstantShape()
                           ? builder.create<fir::AllocMemOp>(loc, type)
                           : builder.create<fir::AllocMemOp>(
                                 loc, type, ".array.expr", llvm::None, shape);
    fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup(
        [bldr, loc, temp]() { bldr->create<fir::FreeMemOp>(loc, temp); });
    mlir::Value shapeOp = genShapeOp(shape);
    return builder.create<fir::ArrayLoadOp>(loc, seqTy, temp, shapeOp,
                                            /*slice=*/mlir::Value{},
                                            llvm::None);
  }

  static fir::ShapeOp genShapeOp(mlir::Location loc, fir::FirOpBuilder &builder,
                                 llvm::ArrayRef<mlir::Value> shape) {
    mlir::IndexType idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> idxShape;
    for (auto s : shape)
      idxShape.push_back(builder.createConvert(loc, idxTy, s));
    auto shapeTy = fir::ShapeType::get(builder.getContext(), idxShape.size());
    return builder.create<fir::ShapeOp>(loc, shapeTy, idxShape);
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

  // An expression with non-zero rank is an array expression.
  template <typename A>
  bool isArray(const A &x) const {
    return x.Rank() != 0;
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
      stmtCtx.finalize(/*popScope=*/true);
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

  // A procedure reference to a Fortran elemental intrinsic procedure.
  CC genElementalIntrinsicProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      llvm::Optional<mlir::Type> retTy,
      const Fortran::evaluate::SpecificIntrinsic &intrinsic) {
    llvm::SmallVector<CC> operands;
    llvm::StringRef name = intrinsic.name;
    const Fortran::lower::IntrinsicArgumentLoweringRules *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    mlir::Location loc = getLoc();
    if (Fortran::lower::intrinsicRequiresCustomOptionalHandling(
            procRef, intrinsic, converter)) {
      using CcPairT = std::pair<CC, llvm::Optional<mlir::Value>>;
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
      auto prepareOtherArg = [&](const Fortran::lower::SomeExpr &expr) {
        PushSemantics(ConstituentSemantics::RefTransparent);
        operands.emplace_back(genElementalArgument(expr), llvm::None);
      };
      Fortran::lower::prepareCustomIntrinsicArgument(
          procRef, intrinsic, retTy, prepareOptionalArg, prepareOtherArg,
          converter);

      fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
      llvm::StringRef name = intrinsic.name;
      return [=](IterSpace iters) -> ExtValue {
        auto getArgument = [&](std::size_t i) -> ExtValue {
          return operands[i].first(iters);
        };
        auto isPresent = [&](std::size_t i) -> llvm::Optional<mlir::Value> {
          return operands[i].second;
        };
        return Fortran::lower::lowerCustomIntrinsic(
            *bldr, loc, name, retTy, isPresent, getArgument, operands.size(),
            getElementCtx());
      };
    }
    /// Otherwise, pre-lower arguments and use intrinsic lowering utility.
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      const auto *expr =
          Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back([=](IterSpace) { return mlir::Value{}; });
      } else if (!argLowering) {
        // No argument lowering instruction, lower by value.
        PushSemantics(ConstituentSemantics::RefTransparent);
        operands.emplace_back(genElementalArgument(*expr));
      } else {
        // Ad-hoc argument lowering handling.
        Fortran::lower::ArgLoweringRule argRules =
            Fortran::lower::lowerIntrinsicArgumentAs(getLoc(), *argLowering,
                                                     dummy.name);
        if (argRules.handleDynamicOptional &&
            Fortran::evaluate::MayBePassedAsAbsentOptional(
                *expr, converter.getFoldingContext())) {
          // Currently, there is not elemental intrinsic that requires lowering
          // a potentially absent argument to something else than a value (apart
          // from character MAX/MIN that are handled elsewhere.)
          if (argRules.lowerAs != Fortran::lower::LowerIntrinsicArgAs::Value)
            TODO(loc, "lowering non trivial optional elemental intrinsic array "
                      "argument");
          PushSemantics(ConstituentSemantics::RefTransparent);
          operands.emplace_back(genarrForwardOptionalArgumentToCall(*expr));
          continue;
        }
        switch (argRules.lowerAs) {
        case Fortran::lower::LowerIntrinsicArgAs::Value: {
          PushSemantics(ConstituentSemantics::RefTransparent);
          operands.emplace_back(genElementalArgument(*expr));
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Addr: {
          // Note: assume does not have Fortran VALUE attribute semantics.
          PushSemantics(ConstituentSemantics::RefOpaque);
          operands.emplace_back(genElementalArgument(*expr));
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Box: {
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genElementalArgument(*expr);
          operands.emplace_back([=](IterSpace iters) {
            return builder.createBox(loc, lambda(iters));
          });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Inquired:
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

  /// Generate a procedure reference. This code is shared for both functions and
  /// subroutines, the difference being reflected by `retTy`.
  CC genProcRef(const Fortran::evaluate::ProcedureRef &procRef,
                llvm::Optional<mlir::Type> retTy) {
    mlir::Location loc = getLoc();
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
      if (ScalarExprLowering::isStatementFunctionCall(procRef))
        fir::emitFatalError(loc, "statement function cannot be elemental");

      TODO(loc, "elemental user defined proc ref");
    }

    // Transformational call.
    // The procedure is called once and produces a value of rank > 0.
    if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
            procRef.proc().GetSpecificIntrinsic()) {
      if (explicitSpaceIsActive() && procRef.Rank() == 0) {
        // Elide any implicit loop iters.
        return [=, &procRef](IterSpace) {
          return ScalarExprLowering{loc, converter, symMap, stmtCtx}
              .genIntrinsicRef(procRef, *intrinsic, retTy);
        };
      }
      return genarr(
          ScalarExprLowering{loc, converter, symMap, stmtCtx}.genIntrinsicRef(
              procRef, *intrinsic, retTy));
    }

    if (explicitSpaceIsActive() && procRef.Rank() == 0) {
      // Elide any implicit loop iters.
      return [=, &procRef](IterSpace) {
        return ScalarExprLowering{loc, converter, symMap, stmtCtx}
            .genProcedureRef(procRef, retTy);
      };
    }
    // In the default case, the call can be hoisted out of the loop nest. Apply
    // the iterations to the result, which may be an array value.
    return genarr(
        ScalarExprLowering{loc, converter, symMap, stmtCtx}.genProcedureRef(
            procRef, retTy));
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
    if (isArray(x) || explicitSpaceIsActive() ||
        isElementalProcWithArrayArgs(x))
      return std::visit([&](const auto &e) { return genarr(e); }, x.u);
    return genScalarAndForwardValue(x);
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  CC genarr(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                             TC2> &x) {
    TODO(getLoc(), "");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    TODO(getLoc(), "");
  }

  template <typename T>
  CC genarr(const Fortran::evaluate::Parentheses<T> &x) {
    TODO(getLoc(), "");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    TODO(getLoc(), "");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    TODO(getLoc(), "");
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    TODO(getLoc(), "");
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
    TODO(getLoc(), "genarr Power<Fortran::evaluate::Type<TC, KIND>>");
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    TODO(getLoc(), "genarr Extremum<Fortran::evaluate::Type<TC, KIND>>");
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    TODO(getLoc(), "genarr RealToIntPower<Fortran::evaluate::Type<TC, KIND>>");
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    TODO(getLoc(), "genarr ComplexConstructor<KIND>");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::Concat<KIND> &x) {
    TODO(getLoc(), "genarr Concat<KIND>");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::SetLength<KIND> &x) {
    TODO(getLoc(), "genarr SetLength<KIND>");
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Constant<A> &x) {
    if (/*explicitSpaceIsActive() &&*/ x.Rank() == 0)
      return genScalarAndForwardValue(x);
    mlir::Location loc = getLoc();
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Type arrTy = converter.genType(toEvExpr(x));
    std::string globalName = Fortran::lower::mangle::mangleArrayLiteral(x);
    fir::GlobalOp global = builder.getNamedGlobal(globalName);
    if (!global) {
      mlir::Type symTy = arrTy;
      mlir::Type eleTy = symTy.cast<fir::SequenceType>().getEleTy();
      // If we have a rank-1 array of integer, real, or logical, then we can
      // create a global array with the dense attribute.
      //
      // The mlir tensor type can only handle integer, real, or logical. It
      // does not currently support nested structures which is required for
      // complex.
      //
      // Also, we currently handle just rank-1 since tensor type assumes
      // row major array ordering. We will need to reorder the dimensions
      // in the tensor type to support Fortran's column major array ordering.
      // How to create this tensor type is to be determined.
      if (x.Rank() == 1 &&
          eleTy.isa<fir::LogicalType, mlir::IntegerType, mlir::FloatType>())
        global = Fortran::lower::createDenseGlobal(
            loc, arrTy, globalName, builder.createInternalLinkage(), true,
            toEvExpr(x), converter);
      // Note: If call to createDenseGlobal() returns 0, then call
      // createGlobalConstant() below.
      if (!global)
        global = builder.createGlobalConstant(
            loc, arrTy, globalName,
            [&](fir::FirOpBuilder &builder) {
              Fortran::lower::StatementContext stmtCtx(
                  /*cleanupProhibited=*/true);
              fir::ExtendedValue result =
                  Fortran::lower::createSomeInitializerExpression(
                      loc, converter, toEvExpr(x), symMap, stmtCtx);
              mlir::Value castTo =
                  builder.createConvert(loc, arrTy, fir::getBase(result));
              builder.create<fir::HasValueOp>(loc, castTo);
            },
            builder.createInternalLinkage());
    }
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    auto seqTy = global.getType().cast<fir::SequenceType>();
    llvm::SmallVector<mlir::Value> extents;
    for (auto extent : seqTy.getShape())
      extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
    if (auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>()) {
      mlir::Value len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                                      charTy.getLen());
      return genarr(fir::CharArrayBoxValue{addr, len, extents});
    }
    return genarr(fir::ArrayBoxValue{addr, extents});
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
        // Array of char with dynamic length parameter. Downcast to an array
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
    return fir::factory::getLlvmMemcpy(builder).getType();
  }

  /// Create a call to the LLVM memcpy intrinsic.
  void createCallMemcpy(llvm::ArrayRef<mlir::Value> args) {
    mlir::Location loc = getLoc();
    mlir::FuncOp memcpyFunc = fir::factory::getLlvmMemcpy(builder);
    mlir::SymbolRefAttr funcSymAttr =
        builder.getSymbolRefAttr(memcpyFunc.getName());
    mlir::FunctionType funcTy = memcpyFunc.getType();
    builder.create<fir::CallOp>(loc, funcTy.getResults(), funcSymAttr, args);
  }

  // Construct code to check for a buffer overrun and realloc the buffer when
  // space is depleted. This is done between each item in the ac-value-list.
  mlir::Value growBuffer(mlir::Value mem, mlir::Value needed,
                         mlir::Value bufferSize, mlir::Value buffSize,
                         mlir::Value eleSz) {
    mlir::Location loc = getLoc();
    mlir::FuncOp reallocFunc = fir::factory::getRealloc(builder);
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
    mlir::FunctionType funcTy = reallocFunc.getType();
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
    llvm::Optional<mlir::Value> charLen;
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
      if (fir::isa_char(seqTy.getEleTy()) && !charLen.hasValue()) {
        charLen = builder.createTemporary(loc, builder.getI64Type());
        mlir::Value castLen =
            builder.createConvert(loc, builder.getI64Type(), fir::getLen(exv));
        builder.create<fir::StoreOp>(loc, castLen, charLen.getValue());
      }
    }
    stmtCtx.finalize(/*popScope=*/true);

    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    mem = loop.getResult(0);
    symMap.popImpliedDoBinding();
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos).getResult()};

    // Convert to extended value.
    if (fir::isa_char(seqTy.getEleTy())) {
      auto len = builder.create<fir::LoadOp>(loc, charLen.getValue());
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
            loc, eleTy, /*typeparams=*/llvm::None, initBuffSz);
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
    llvm::Optional<mlir::Value> charLen;
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
      if (fir::isa_char(seqTy.getEleTy()) && !charLen.hasValue()) {
        charLen = builder.createTemporary(loc, builder.getI64Type());
        mlir::Value castLen =
            builder.createConvert(loc, builder.getI64Type(), fir::getLen(exv));
        builder.create<fir::StoreOp>(loc, castLen, charLen.getValue());
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
      if (charLen.hasValue()) {
        auto len = builder.create<fir::LoadOp>(loc, charLen.getValue());
        return genarr(fir::CharArrayBoxValue{mem, len, extents});
      }
      return genarr(fir::CharArrayBoxValue{mem, zero, extents});
    }
    return genarr(fir::ArrayBoxValue{mem, extents});
  }

  CC genarr(const Fortran::evaluate::ImpliedDoIndex &) {
    TODO(getLoc(), "genarr ImpliedDoIndex");
  }

  CC genarr(const Fortran::evaluate::TypeParamInquiry &x) {
    TODO(getLoc(), "genarr TypeParamInquiry");
  }

  CC genarr(const Fortran::evaluate::DescriptorInquiry &x) {
    TODO(getLoc(), "genarr DescriptorInquiry");
  }

  CC genarr(const Fortran::evaluate::StructureConstructor &x) {
    TODO(getLoc(), "genarr StructureConstructor");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::Not<KIND> &x) {
    TODO(getLoc(), "genarr Not");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::LogicalOperation<KIND> &x) {
    TODO(getLoc(), "genarr LogicalOperation");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    TODO(getLoc(), "genarr Relational Integer");
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Character, KIND>> &x) {
    TODO(getLoc(), "genarr Relational Character");
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    TODO(getLoc(), "genarr Relational Real");
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    TODO(getLoc(), "genarr Relational Complex");
  }
  CC genarr(
      const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &r) {
    TODO(getLoc(), "genarr Relational SomeType");
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Designator<A> &des) {
    ComponentPath components(des.Rank() > 0);
    return std::visit([&](const auto &x) { return genarr(x, components); },
                      des.u);
  }

  template <typename T>
  CC genarr(const Fortran::evaluate::FunctionRef<T> &funRef) {
    // Note that it's possible that the function being called returns either an
    // array or a scalar.  In the first case, use the element type of the array.
    return genProcRef(
        funRef, fir::unwrapSequenceType(converter.genType(toEvExpr(funRef))));
  }

  template <typename A>
  CC genImplicitArrayAccess(const A &x, ComponentPath &components) {
    components.reversePath.push_back(ImplicitSubscripts{});
    ExtValue exv = asScalarRef(x);
    // lowerPath(exv, components);
    auto lambda = genarr(exv, components);
    return [=](IterSpace iters) { return lambda(components.pc(iters)); };
  }
  CC genImplicitArrayAccess(const Fortran::evaluate::NamedEntity &x,
                            ComponentPath &components) {
    if (x.IsSymbol())
      return genImplicitArrayAccess(x.GetFirstSymbol(), components);
    return genImplicitArrayAccess(x.GetComponent(), components);
  }

  template <typename A>
  CC genAsScalar(const A &x) {
    mlir::Location loc = getLoc();
    if (isProjectedCopyInCopyOut()) {
      return [=, &x, builder = &converter.getFirOpBuilder()](
                 IterSpace iters) -> ExtValue {
        ExtValue exv = asScalarRef(x);
        mlir::Value val = fir::getBase(exv);
        mlir::Type eleTy = fir::unwrapRefType(val.getType());
        if (isAdjustedArrayElementType(eleTy)) {
          if (fir::isa_char(eleTy)) {
            TODO(getLoc(), "assignment of character type");
          } else if (fir::isa_derived(eleTy)) {
            TODO(loc, "assignment of derived type");
          } else {
            fir::emitFatalError(loc, "array type not expected in scalar");
          }
        } else {
          builder->create<fir::StoreOp>(loc, iters.getElement(), val);
        }
        return exv;
      };
    }
    return [=, &x](IterSpace) { return asScalar(x); };
  }

  CC genarr(const Fortran::semantics::Symbol &x, ComponentPath &components) {
    if (explicitSpaceIsActive()) {
      TODO(getLoc(), "genarr Symbol explicitSpace");
    } else {
      return genImplicitArrayAccess(x, components);
    }
  }

  CC genarr(const Fortran::evaluate::Component &x, ComponentPath &components) {
    TODO(getLoc(), "genarr Component");
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
      TODO(getLoc(), "genarr ArrayRef explicitSpace");
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
    return x.IsSymbol() ? genarr(x.GetFirstSymbol(), components)
                        : genarr(x.GetComponent(), components);
  }

  CC genarr(const Fortran::evaluate::DataRef &x, ComponentPath &components) {
    return std::visit([&](const auto &v) { return genarr(v, components); },
                      x.u);
  }

  bool pathIsEmpty(const ComponentPath &components) {
    return components.reversePath.empty();
  }

  /// Given an optional fir.box, returns an fir.box that is the original one if
  /// it is present and it otherwise an unallocated box.
  /// Absent fir.box are implemented as a null pointer descriptor. Generated
  /// code may need to unconditionally read a fir.box that can be absent.
  /// This helper allows creating a fir.box that can be read in all cases
  /// outside of a fir.if (isPresent) region. However, the usages of the value
  /// read from such box should still only be done in a fir.if(isPresent).
  static fir::ExtendedValue
  absentBoxToUnalllocatedBox(fir::FirOpBuilder &builder, mlir::Location loc,
                             const fir::ExtendedValue &exv,
                             mlir::Value isPresent) {
    mlir::Value box = fir::getBase(exv);
    mlir::Type boxType = box.getType();
    assert(boxType.isa<fir::BoxType>() && "argument must be a fir.box");
    mlir::Value emptyBox =
        fir::factory::createUnallocatedBox(builder, loc, boxType, llvm::None);
    auto safeToReadBox =
        builder.create<mlir::arith::SelectOp>(loc, isPresent, box, emptyBox);
    return fir::substBase(exv, safeToReadBox);
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
      exv = absentBoxToUnalllocatedBox(builder, loc, exv, isPresent);
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
      for (unsigned i = 1, end = triples.size(); i < end; i += 3)
        if (!mlir::isa_and_nonnull<fir::UndefOp>(triples[i].getDefiningOp()))
          shape.push_back(fir::SequenceType::getUnknownExtent());
      return fir::SequenceType::get(shape, seqTy.getEleTy());
    }
    // not sliced, so no change in rank
    return arrTy;
  }

  CC genarr(const Fortran::evaluate::ComplexPart &x,
            ComponentPath &components) {
    TODO(getLoc(), "genarr ComplexPart");
  }

  CC genarr(const Fortran::evaluate::StaticDataObject::Pointer &,
            ComponentPath &components) {
    TODO(getLoc(), "genarr StaticDataObject::Pointer");
  }

  /// Substrings (see 9.4.1)
  CC genarr(const Fortran::evaluate::Substring &x, ComponentPath &components) {
    TODO(getLoc(), "genarr Substring");
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
        slice = builder.create<fir::SliceOp>(loc, components.trips,
                                             components.suffixComponents,
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
      auto boxTy = fir::BoxType::get(reduceTy);
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
      mlir::Value embox =
          memref.getType().isa<fir::BoxType>()
              ? builder.create<fir::ReboxOp>(loc, boxTy, memref, shape, slice)
                    .getResult()
              : builder
                    .create<fir::EmboxOp>(loc, boxTy, memref, shape, slice,
                                          fir::getTypeParams(extMemref))
                    .getResult();
      return [=](IterSpace) -> ExtValue { return fir::BoxValue(embox); };
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
      auto lambda = ccStoreToDest.hasValue()
                        ? ccStoreToDest.getValue()
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
    mlir::Operation::operand_range arrLdTypeParams = arrLoad.getTypeparams();
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

private:
  void determineShapeOfDest(const fir::ExtendedValue &lhs) {
    destShape = fir::factory::getExtents(builder, getLoc(), lhs);
  }

  void determineShapeOfDest(const Fortran::lower::SomeExpr &lhs) {
    if (!destShape.empty())
      return;
    // if (explicitSpaceIsActive() && determineShapeWithSlice(lhs))
    //   return;
    mlir::Type idxTy = builder.getIndexType();
    mlir::Location loc = getLoc();
    if (std::optional<Fortran::evaluate::ConstantSubscripts> constantShape =
            Fortran::evaluate::GetConstantExtents(converter.getFoldingContext(),
                                                  lhs))
      for (Fortran::common::ConstantSubscript extent : *constantShape)
        destShape.push_back(builder.createIntegerConstant(loc, idxTy, extent));
  }

  ExtValue lowerArrayExpression(const Fortran::lower::SomeExpr &exp) {
    mlir::Type resTy = converter.genType(exp);
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
            destination.getTypeparams());
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
      mlir::Value ele = builder.createConvert(loc, eleTy, fir::getBase(exv));
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
    auto lambda = ccStoreToDest.hasValue()
                      ? ccStoreToDest.getValue()
                      : defaultStoreToDestination(/*substring=*/nullptr);
    mlir::Value updVal = fir::getBase(lambda(iterSpace));
    finalizeElementCtx();
    builder.create<fir::ResultOp>(loc, updVal);
    builder.restoreInsertionPoint(insPt);
    return abstractArrayExtValue(iterSpace.outerResult());
  }

  /// Get the shape from an ArrayOperand. The shape of the array is adjusted if
  /// the array was sliced.
  llvm::SmallVector<mlir::Value> getShape(ArrayOperand array) {
    // if (array.slice)
    //   return computeSliceShape(array.slice);
    if (array.memref.getType().isa<fir::BoxType>())
      return fir::factory::readExtents(builder, getLoc(),
                                       fir::BoxValue{array.memref});
    std::vector<mlir::Value, std::allocator<mlir::Value>> extents =
        fir::factory::getExtents(array.shape);
    return {extents.begin(), extents.end()};
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
    fir::emitFatalError(getLoc(),
                        "failed to compute the array expression shape");
  }

  bool explicitSpaceIsActive() const {
    return explicitSpace && explicitSpace->isActive();
  }

  bool implicitSpaceHasMasks() const {
    return implicitSpace && !implicitSpace->empty();
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
        explicitSpace(expSpace->isActive() ? expSpace : nullptr),
        implicitSpace(impSpace->empty() ? nullptr : impSpace), semant{sem} {
    // Generate any mask expressions, as necessary. This is the compute step
    // that creates the effective masks. See 10.2.3.2 in particular.
    // genMasks();
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

  inline bool isCustomCopyInCopyOut() {
    return semant == ConstituentSemantics::CustomCopyInCopyOut;
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

  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  bool elementCtx = false;
  Fortran::lower::SymMap &symMap;
  /// The continuation to generate code to update the destination.
  llvm::Optional<CC> ccStoreToDest;
  llvm::Optional<std::function<void(llvm::ArrayRef<mlir::Value>)>> ccPrelude;
  llvm::Optional<std::function<fir::ArrayLoadOp(llvm::ArrayRef<mlir::Value>)>>
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
  // Can the array expression be evaluated in any order?
  // Will be set to false if any of the expression parts prevent this.
  bool unordered = true;
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.genval(expr);
}

fir::GlobalOp Fortran::lower::createDenseGlobal(
    mlir::Location loc, mlir::Type symTy, llvm::StringRef globalName,
    mlir::StringAttr linkage, bool isConst,
    const Fortran::lower::SomeExpr &expr,
    Fortran::lower::AbstractConverter &converter) {

  Fortran::lower::StatementContext stmtCtx(/*prohibited=*/true);
  Fortran::lower::SymMap emptyMap;
  InitializerData initData(/*genRawVals=*/true);
  ScalarExprLowering sel(loc, converter, emptyMap, stmtCtx,
                         /*initializer=*/&initData);
  sel.genval(expr);

  size_t sz = initData.rawVals.size();
  llvm::ArrayRef<mlir::Attribute> ar = {initData.rawVals.data(), sz};

  mlir::RankedTensorType tensorTy;
  auto &builder = converter.getFirOpBuilder();
  mlir::Type iTy = initData.rawType;
  if (!iTy)
    return 0; // array extent is probably 0 in this case, so just return 0.
  tensorTy = mlir::RankedTensorType::get(sz, iTy);
  auto init = mlir::DenseElementsAttr::get(tensorTy, ar);
  return builder.createGlobal(loc, symTy, globalName, linkage, init, isConst);
}

fir::ExtendedValue Fortran::lower::createSomeInitializerExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  InitializerData initData; // needed for initializations
  return ScalarExprLowering{loc, converter, symMap, stmtCtx,
                            /*initializer=*/&initData}
      .genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(expr);
}

fir::ExtendedValue Fortran::lower::createInitializerAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  InitializerData init;
  return ScalarExprLowering(loc, converter, symMap, stmtCtx, &init).gen(expr);
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

fir::ExtendedValue Fortran::lower::createBoxValue(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  if (expr.Rank() > 0 && Fortran::evaluate::IsVariable(expr) &&
      !Fortran::evaluate::HasVectorSubscript(expr))
    return Fortran::lower::createSomeArrayBox(converter, expr, symMap, stmtCtx);
  fir::ExtendedValue addr = Fortran::lower::createSomeExtendedAddress(
      loc, converter, expr, symMap, stmtCtx);
  return fir::BoxValue(converter.getFirOpBuilder().createBox(loc, addr));
}

mlir::Value Fortran::lower::createSubroutineCall(
    AbstractConverter &converter, const evaluate::ProcedureRef &call,
    SymMap &symMap, StatementContext &stmtCtx) {
  mlir::Location loc = converter.getCurrentLocation();

  // Simple subroutine call, with potential alternate return.
  auto res = Fortran::lower::createSomeExtendedExpression(
      loc, converter, toEvExpr(call), symMap, stmtCtx);
  return fir::getBase(res);
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
    const fir::ExtendedValue &rhs, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "onto array: " << lhs << '\n';
             llvm::dbgs() << "assign expression: " << rhs << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
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

fir::ExtendedValue Fortran::lower::createSomeArrayTempValue(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  return ArrayExprLowering::lowerNewArrayExpression(converter, symMap, stmtCtx,
                                                    expr);
}

mlir::Value Fortran::lower::genMaxWithZero(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value value) {
  mlir::Value zero = builder.createIntegerConstant(loc, value.getType(), 0);
  if (mlir::Operation *definingOp = value.getDefiningOp())
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(definingOp))
      if (auto intAttr = cst.getValue().dyn_cast<mlir::IntegerAttr>())
        return intAttr.getInt() < 0 ? zero : value;
  return Fortran::lower::genMax(builder, loc,
                                llvm::SmallVector<mlir::Value>{value, zero});
}
