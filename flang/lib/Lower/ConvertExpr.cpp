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
#include "ConvertVariable.h"
#include "MaskExpr.h"
#include "RTBuilder.h"
#include "StatementContext.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/CharacterRuntime.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "flang-lower-expr"

//===----------------------------------------------------------------------===//
// The composition and structure of Fortran::evaluate::Expr is defined in the
// various header files in include/flang/Evaluate. You are referred there for
// more information on these data structures. Generally speaking, these data
// structures are a strongly typed family of abstract data types that, composed
// as trees, describe the syntax of Fortran expressions.
//
// This part of the bridge can traverse these tree structures and lower them to
// the correct FIR representation in SSA form.
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
  // Scalars : let `v` be the location in memory of a variable with value `x`
  DataValue, // refers to the value `x`
  DataAddr,  // refers to the address `v`
  BoxValue,  // refers to a box value containing `v`
  BoxAddr,   // refers to the address of a box value containing `v`

  // Arrays : let `a` be the location in memory of a sequence of value `[xs]`
  RefTransparent, // refers to the value `[xs]`
  ByValueArg, // refers to an ephemeral address `t` containing a value `x` which
              // is the i-th value in `[xs]` (15.5.2.3.p7 note 2)
  CopyInCopyOut, // refers to the merge of `[xs]` with another value `[ys]`,
                 // which is written into `a`.
  ProjectedCopyInCopyOut, // similar to CopyInCopyOut but the variable `v` may
                          // itself be a transient projection (rather than a
                          // whole array).
  RefOpaque // refers to the address `a+i`, the i-th element of `a`
};

/// Convert parser's INTEGER relational operators to MLIR.  TODO: using
/// unordered, but we may want to cons ordered in certain situation.
static mlir::CmpIPredicate
translateRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::CmpIPredicate::slt;
  case Fortran::common::RelationalOperator::LE:
    return mlir::CmpIPredicate::sle;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::CmpIPredicate::eq;
  case Fortran::common::RelationalOperator::NE:
    return mlir::CmpIPredicate::ne;
  case Fortran::common::RelationalOperator::GT:
    return mlir::CmpIPredicate::sgt;
  case Fortran::common::RelationalOperator::GE:
    return mlir::CmpIPredicate::sge;
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
static mlir::CmpFPredicate
translateFloatRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::CmpFPredicate::OLT;
  case Fortran::common::RelationalOperator::LE:
    return mlir::CmpFPredicate::OLE;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::CmpFPredicate::OEQ;
  case Fortran::common::RelationalOperator::NE:
    return mlir::CmpFPredicate::UNE;
  case Fortran::common::RelationalOperator::GT:
    return mlir::CmpFPredicate::OGT;
  case Fortran::common::RelationalOperator::GE:
    return mlir::CmpFPredicate::OGE;
  }
  llvm_unreachable("unhandled REAL relational operator");
}

/// Clone subexpression and wrap it as a generic `Fortran::evaluate::Expr`.
template <typename A>
Fortran::evaluate::Expr<Fortran::evaluate::SomeType> toEvExpr(const A &x) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x));
}

/// Lower `opt` (from front-end shape analysis) to MLIR. If `opt` is `nullopt`
/// then issue an error.
static mlir::Value
convertOptExtentExpr(Fortran::lower::AbstractConverter &converter,
                     Fortran::lower::StatementContext &stmtCtx,
                     const Fortran::evaluate::MaybeExtentExpr &opt) {
  auto loc = converter.getCurrentLocation();
  if (!opt.has_value())
    fir::emitFatalError(loc, "shape analysis failed to return an expression");
  auto e = toEvExpr(*opt);
  return fir::getBase(converter.genExprValue(&e, stmtCtx, loc));
}

/// Return the extended value for a component of a derived type instance given
/// the extended value \p obj of the derived type instance and the address of
/// the component.
static fir::ExtendedValue
componentToExtendedValue(Fortran::lower::FirOpBuilder &builder,
                         mlir::Location loc, const fir::ExtendedValue &obj,
                         mlir::Value component) {
  auto fieldTy = fir::dyn_cast_ptrEleTy(component.getType());
  if (fieldTy.dyn_cast<fir::BoxType>())
    TODO(loc, "lower pointer/allocatable component ref");
  llvm::SmallVector<mlir::Value> extents;
  if (auto seqTy = fieldTy.dyn_cast<fir::SequenceType>()) {
    fieldTy = seqTy.getEleTy();
    auto idxTy = builder.getIndexType();
    for (auto extent : seqTy.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        TODO(loc, "array component shape depending on length parameters");
      extents.emplace_back(builder.createIntegerConstant(loc, idxTy, extent));
    }
  }
  if (auto charTy = fieldTy.dyn_cast<fir::CharacterType>()) {
    auto cstLen = charTy.getLen();
    if (cstLen == fir::CharacterType::unknownLen())
      TODO(loc, "get character component length from length type parameters");
    auto len = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), cstLen);
    if (!extents.empty())
      return fir::CharArrayBoxValue{component, len, extents};
    return fir::CharBoxValue{component, len};
  }
  if (auto recordTy = fieldTy.dyn_cast<fir::RecordType>())
    if (recordTy.getNumLenParams() != 0)
      TODO(loc,
           "lower component ref that is a derived type with length parameter");
  if (!extents.empty())
    return fir::ArrayBoxValue{component, extents};
  return component;
}

/// Given the address of an array element and the ExtendedValue describing the
/// array, returns the ExtendedValue describing the array element. The purpose
/// is to propagate the length parameters of the array to the element.
static fir::ExtendedValue
arrayElementToExtendedValue(Fortran::lower::FirOpBuilder &builder,
                            mlir::Location loc, const fir::ExtendedValue &array,
                            mlir::Value element) {
  return array.match(
      [&](const fir::CharBoxValue &cb) -> fir::ExtendedValue {
        return cb.clone(element);
      },
      [&](const fir::CharArrayBoxValue &bv) -> fir::ExtendedValue {
        return bv.cloneElement(element);
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        if (box.isCharacter()) {
          auto len = Fortran::lower::readCharLen(builder, loc, box);
          return fir::CharBoxValue{element, len};
        }
        if (box.isDerived())
          TODO(loc, "get length parameters from derived type BoxValue");
        return element;
      },
      [&](const auto &) -> fir::ExtendedValue { return element; });
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
                              bool initializer = false)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap},
        inInitializer{initializer} {}

  ExtValue genExtAddr(const Fortran::lower::SomeExpr &expr) {
    return gen(expr);
  }

  /// Lower `expr` to be passed as an argument. The expression is passed by
  /// reference.
  ExtValue genExtArg(const Fortran::lower::SomeExpr &expr, bool mayUseBox) {
    bool saveUseBoxArg = useBoxArg;
    useBoxArg = mayUseBox;
    auto result = gen(expr);
    useBoxArg = saveUseBoxArg;
    return result;
  }

  ExtValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  fir::MutableBoxValue
  genMutableBoxValue(const Fortran::lower::SomeExpr &expr) {
    // TODO: expr may not be a scalar variable here, it could be an allocatable
    // or pointer derived type component, and this will fail.
    auto *sym = Fortran::evaluate::UnwrapWholeSymbolDataRef(expr);
    if (!sym)
      TODO(getLoc(), "trying to get descriptor address of an expression that "
                     "is not a simple scalar variable");
    return symMap.lookupSymbol(*sym).match(
        [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &boxAddr) {
          return boxAddr;
        },
        [&](auto &) -> fir::MutableBoxValue {
          fir::emitFatalError(getLoc(),
                              "symbol was not lowered to MutableBoxValue");
        });
  }

  mlir::Location getLoc() { return location; }

  template <typename A>
  mlir::Value genunbox(const A &expr) {
    auto e = genval(expr);
    if (auto *r = e.getUnboxed())
      return *r;
    fir::emitFatalError(getLoc(), "unboxed expression expected");
  }

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    auto type = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
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
    auto fltTy = Fortran::lower::convertReal(context, KIND);
    return builder.createRealConstant(getLoc(), fltTy, value);
  }

  mlir::Type getSomeKindInteger() { return builder.getIndexType(); }

  mlir::FuncOp getFunction(llvm::StringRef name, mlir::FunctionType funTy) {
    if (auto func = builder.getNamedFunction(name))
      return func;
    return builder.createFunction(getLoc(), name, funTy);
  }

  template <typename OpTy>
  mlir::Value createCompareOp(mlir::CmpIPredicate pred, const ExtValue &left,
                              const ExtValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::CmpIPredicate pred) {
    return createCompareOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  template <typename OpTy>
  mlir::Value createFltCmpOp(mlir::CmpFPredicate pred, const ExtValue &left,
                             const ExtValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  /// Create a call to the runtime to compare two CHARACTER values.
  /// Precondition: This assumes that the two values have `fir.boxchar` type.
  mlir::Value createCharCompare(mlir::CmpIPredicate pred, const ExtValue &left,
                                const ExtValue &right) {
    return Fortran::lower::genCharCompare(builder, getLoc(), pred, left, right);
  }

  template <typename A>
  mlir::Value createCharCompare(const A &ex, mlir::CmpIPredicate pred) {
    return createCharCompare(pred, genval(ex.left()), genval(ex.right()));
  }

  Fortran::lower::SymbolBox
  genAllocatableOrPointerUnbox(const fir::MutableBoxValue &box) {
    return Fortran::lower::genMutableBoxRead(builder, getLoc(), box);
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  ExtValue gen(Fortran::semantics::SymbolRef sym) {
    if (auto val = symMap.lookupSymbol(sym))
      return val.match(
          [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &boxAddr) {
            return genAllocatableOrPointerUnbox(boxAddr).toExtendedValue();
          },
          [&val](auto &) { return val.toExtendedValue(); });
    llvm_unreachable("all symbols should be in the map");
    auto addr = builder.createTemporary(getLoc(), converter.genType(sym),
                                        toStringRef(sym->name()));
    symMap.addSymbol(sym, addr);
    return addr;
  }

  /// Generate a load of a value from an address.
  ExtValue genLoad(const ExtValue &addr) {
    auto loc = getLoc();
    return addr.match(
        [](const fir::CharBoxValue &box) -> ExtValue { return box; },
        [&](const fir::UnboxedValue &v) -> ExtValue {
          return builder.create<fir::LoadOp>(loc, fir::getBase(v));
        },
        [&](const auto &v) -> ExtValue {
          TODO(getLoc(), "loading array or descriptor");
        });
  }

  ExtValue genval(Fortran::semantics::SymbolRef sym) {
    auto loc = getLoc();
    auto var = gen(sym);
    if (auto *s = var.getUnboxed())
      if (fir::isReferenceLike(s->getType())) {
        // A function with multiple entry points returning different types
        // tags all result variables with one of the largest types to allow
        // them to share the same storage.  A reference to a result variable
        // of one of the other types requires conversion to the actual type.
        auto addr = *s;
        if (Fortran::semantics::IsFunctionResult(sym)) {
          auto resultType = converter.genType(*sym);
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
    if (const auto *intrinsic = proc.GetSpecificIntrinsic()) {
      auto signature = Fortran::lower::translateSignature(proc, converter);
      // Intrinsic lowering is based on the generic name, so retrieve it here in
      // case it is different from the specific name. The type of the specific
      // intrinsic is retained in the signature.
      auto genericName =
          converter.getFoldingContext().intrinsics().GetGenericIntrinsicName(
              intrinsic->name);
      auto symbolRefAttr =
          Fortran::lower::getUnrestrictedIntrinsicSymbolRefAttr(
              builder, getLoc(), genericName, signature);
      mlir::Value funcPtr =
          builder.create<fir::AddrOfOp>(getLoc(), signature, symbolRefAttr);
      return funcPtr;
    }
    const auto *symbol = proc.GetSymbol();
    assert(symbol && "expected symbol in ProcedureDesignator");
    if (Fortran::semantics::IsDummy(*symbol)) {
      auto val = symMap.lookupSymbol(*symbol);
      assert(val && "Dummy procedure not in symbol map");
      return val.getAddr();
    }
    auto name = converter.mangleName(*symbol);
    auto func = Fortran::lower::getOrDeclareFunction(name, proc, converter);
    mlir::Value funcPtr = builder.create<fir::AddrOfOp>(
        getLoc(), func.getType(), builder.getSymbolRefAttr(name));
    return funcPtr;
  }
  ExtValue genval(const Fortran::evaluate::NullPointer &) {
    return builder.createNullConstant(getLoc());
  }

  /// A structure constructor is lowered two ways. In an initializer context,
  /// the entire structure must be constant, so the aggregate value is
  /// constructed inline. This allows it to be the body of a GlobalOp.
  /// Otherwise, the structure constructor is in an expression. In that case, a
  /// temporary object is constructed in the stack frame of the procedure.
  ExtValue genval(const Fortran::evaluate::StructureConstructor &ctor) {
    auto loc = getLoc();
    auto ty = translateSomeExprToFIRType(converter, toEvExpr(ctor));
    auto recTy = ty.cast<fir::RecordType>();
    auto fieldTy = fir::FieldType::get(ty.getContext());
    mlir::Value res = inInitializer ? builder.create<fir::UndefOp>(loc, recTy)
                                    : builder.createTemporary(loc, recTy);
    for (auto [sym, expr] : ctor.values()) {
      auto val = genval(expr.value());
      auto valTy = fir::getBase(val).getType();
      auto name = toStringRef(sym->name());
      // FIXME: type parameters must come from the derived-type-spec
      mlir::Value field = builder.create<fir::FieldIndexOp>(
          loc, fieldTy, name, ty,
          /*typeParams=*/mlir::ValueRange{} /*TODO*/);
      if (fir::isa_ref_type(valTy)) {
        assert(!inInitializer && "expecting a constant value");
        auto fldType = fir::dyn_cast_ptrEleTy(valTy);
        if (fir::isa_char(fldType)) {
          auto fldCharTy = fldType.cast<fir::CharacterType>();
          if (!fldCharTy.hasConstantLen())
            TODO(loc, "CHARACTER LEN type parameter not constant");
          mlir::Value field = builder.create<fir::FieldIndexOp>(
              loc, fieldTy, name, fldType,
              /*typeParams=*/mlir::ValueRange{} /*TODO*/);
          auto fldRefTy = builder.getRefType(fldType);
          auto lenVal = builder.createIntegerConstant(loc, builder.getI64Type(),
                                                      fldCharTy.getLen());
          mlir::Value from = builder.create<fir::CoordinateOp>(
              loc, fldRefTy, fir::getBase(val), field);
          ExtValue fromPtr{fir::CharBoxValue{from, lenVal}};
          mlir::Value to = builder.create<fir::CoordinateOp>(
              loc, fldRefTy, fir::getBase(res), field);
          ExtValue toPtr{fir::CharBoxValue{to, lenVal}};
          Fortran::lower::CharacterExprHelper{builder, loc}.createAssign(
              toPtr, fromPtr);
          continue;
        }
      }
      if (inInitializer) {
        auto memTy = recTy.getType(name);
        auto castVal = builder.createConvert(loc, memTy, fir::getBase(val));
        res =
            builder.create<fir::InsertValueOp>(loc, recTy, res, castVal, field);
      } else {
        auto coorTy = builder.getRefType(recTy.getType(name));
        auto coor = builder.create<fir::CoordinateOp>(loc, coorTy,
                                                      fir::getBase(res), field);
        builder.create<fir::StoreOp>(loc, fir::getBase(val), coor);
      }
    }
    return inInitializer ? res : builder.create<fir::LoadOp>(loc, res);
  }

  /// Lowering of an <i>ac-do-variable</i>, which is not a Symbol.
  ExtValue genval(const Fortran::evaluate::ImpliedDoIndex &var) {
    return converter.impliedDoBinding(toStringRef(var.name));
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    auto exv = desc.base().IsSymbol() ? gen(desc.base().GetLastSymbol())
                                      : gen(desc.base().GetComponent());
    auto idxTy = builder.getIndexType();
    auto loc = getLoc();
    auto castResult = [&](mlir::Value v) {
      using ResTy = Fortran::evaluate::DescriptorInquiry::Result;
      return builder.createConvert(
          loc, converter.genType(ResTy::category, ResTy::kind), v);
    };
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      return castResult(Fortran::lower::readCharLen(builder, loc, exv));
    case Fortran::evaluate::DescriptorInquiry::Field::LowerBound:
      return castResult(Fortran::lower::readLowerBound(
          builder, loc, exv, desc.dimension(),
          builder.createIntegerConstant(loc, idxTy, 1)));
    case Fortran::evaluate::DescriptorInquiry::Field::Extent:
      return castResult(
          Fortran::lower::readExtent(builder, loc, exv, desc.dimension()));
    case Fortran::evaluate::DescriptorInquiry::Field::Rank:
      TODO(loc, "rank inquiry on assumed rank");
    case Fortran::evaluate::DescriptorInquiry::Field::Stride:
      // So far the front end does not generate this inquiry.
      TODO(loc, "Stride inquiry");
    }
    llvm_unreachable("unknown descriptor inquiry");
  }

  ExtValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "type parameter inquiry");
  }

  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return Fortran::lower::ComplexExprHelper{builder, getLoc()}
        .extractComplexPart(cplx, isImagPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    return extractComplexPart(genunbox(part.left()), part.isImaginaryPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    auto input = genunbox(op.left());
    // Like LLVM, integer negation is the binary op "0 - value"
    auto zero = genIntegerConstant<KIND>(builder.getContext(), 0);
    return builder.create<mlir::SubIOp>(getLoc(), zero, input);
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return builder.create<fir::NegfOp>(getLoc(), genunbox(op.left()));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return builder.create<fir::NegcOp>(getLoc(), genunbox(op.left()));
  }

  template <typename OpTy>
  mlir::Value createBinaryOp(const ExtValue &left, const ExtValue &right) {
    assert(fir::isUnboxedValue(left) && fir::isUnboxedValue(right));
    auto lhs = fir::getBase(left);
    auto rhs = fir::getBase(right);
    assert(lhs.getType() == rhs.getType() && "types must be the same");
    return builder.create<OpTy>(getLoc(), lhs, rhs);
  }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    return createBinaryOp<OpTy>(genval(ex.left()), genval(ex.right()));
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  ExtValue genval(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type< \
                      Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) { \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::AddIOp)
  GENBIN(Add, Real, mlir::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::SubIOp)
  GENBIN(Subtract, Real, mlir::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::MulIOp)
  GENBIN(Multiply, Real, mlir::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::SignedDivIOp)
  GENBIN(Divide, Real, mlir::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    return Fortran::lower::ComplexExprHelper{builder, getLoc()}.createComplex(
        KIND, genunbox(op.left()), genunbox(op.right()));
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    auto lhs = genval(op.left());
    auto rhs = genval(op.right());
    auto *lhsChar = lhs.getCharBox();
    auto *rhsChar = rhs.getCharBox();
    if (lhsChar && rhsChar)
      return Fortran::lower::CharacterExprHelper{builder, getLoc()}
          .createConcatenate(*lhsChar, *rhsChar);
    TODO(getLoc(), "character array concatenate");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    switch (op.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return Fortran::lower::genMax(builder, getLoc(),
                                    llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Less:
      return Fortran::lower::genMin(builder, getLoc(),
                                    llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Equal:
      llvm_unreachable("Equal is not a valid ordering in this context");
    }
    llvm_unreachable("unknown ordering");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::SetLength<KIND> &) {
    TODO(getLoc(), "evaluate::SetLength lowering");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    return createCompareOp<mlir::CmpIOp>(op, translateRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return createFltCmpOp<fir::CmpfOp>(op, translateFloatRelational(op.opr));
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
    auto ty = converter.genType(TC1, KIND);
    auto operand = genunbox(convert.left());
    return builder.convertWithSemantics(getLoc(), ty, operand);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    auto input = genval(op.left());
    auto base = fir::getBase(input);
    mlir::Value newBase =
        builder.create<fir::NoReassocOp>(getLoc(), base.getType(), base);
    return fir::substBase(input, newBase);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Not<KIND> &op) {
    auto logical = genunbox(op.left());
    auto one = genBoolConstant(true);
    auto val = builder.createConvert(getLoc(), builder.getI1Type(), logical);
    return builder.create<mlir::XOrOp>(getLoc(), val, one);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    auto i1Type = builder.getI1Type();
    auto slhs = genunbox(op.left());
    auto srhs = genunbox(op.right());
    auto lhs = builder.createConvert(getLoc(), i1Type, slhs);
    auto rhs = builder.createConvert(getLoc(), i1Type, srhs);
    switch (op.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryOp<mlir::AndOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryOp<mlir::OrOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareOp<mlir::CmpIOp>(mlir::CmpIPredicate::eq, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareOp<mlir::CmpIOp>(mlir::CmpIPredicate::ne, lhs, rhs);
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
  /// Convert a ascii scalar literal CHARACTER to IR. (specialization)
  ExtValue
  genAsciiScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                        Fortran::common::TypeCategory::Character, 1>> &value,
                    int64_t len) {
    assert(value.size() == static_cast<std::uint64_t>(len));
    // Outline character constant in ro data if it is not in an initializer.
    if (!inInitializer)
      return Fortran::lower::createStringLiteral(builder, getLoc(), value);
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    auto stringLit = builder.createStringLitOp(getLoc(), value);
    auto lenp = builder.createIntegerConstant(
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
    auto type = fir::CharacterType::get(builder.getContext(), KIND, len);
    auto consLit = [&]() -> fir::StringLitOp {
      auto context = builder.getContext();
      std::int64_t size = static_cast<std::int64_t>(value.size());
      auto shape = mlir::VectorType::get(
          llvm::ArrayRef<std::int64_t>{size},
          mlir::IntegerType::get(builder.getContext(), sizeof(ET) * 8));
      auto strAttr = mlir::DenseElementsAttr::get(
          shape, llvm::ArrayRef<ET>{value.data(), value.size()});
      auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), context);
      mlir::NamedAttribute dataAttr(valTag, strAttr);
      auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), context);
      mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
      llvm::SmallVector<mlir::NamedAttribute> attrs{dataAttr, sizeAttr};
      return builder.create<fir::StringLitOp>(
          getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
    };

    auto lenp = builder.createIntegerConstant(
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
        Fortran::lower::uniqueCGIdent("cl", (const char *)value.c_str());
    auto global = builder.getNamedGlobal(globalName);
    if (!global)
      global = builder.createGlobalConstant(
          getLoc(), type, globalName,
          [&](Fortran::lower::FirOpBuilder &builder) {
            auto str = consLit();
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
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    auto idxTy = builder.getIndexType();
    for (auto [lb, extent] : llvm::zip(con.lbounds(), con.shape())) {
      lbounds.push_back(builder.createIntegerConstant(getLoc(), idxTy, lb - 1));
      extents.push_back(builder.createIntegerConstant(getLoc(), idxTy, extent));
    }
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      fir::SequenceType::Shape shape;
      shape.append(con.shape().begin(), con.shape().end());
      auto chTy = converter.genType(TC, KIND, {con.LEN()});
      auto arrayTy = fir::SequenceType::get(shape, chTy);
      mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
      Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
      do {
        auto charVal =
            fir::getBase(genScalarLit<KIND>(con.At(subscripts), con.LEN()));
        llvm::SmallVector<mlir::Value> idx;
        for (auto [dim, lb] : llvm::zip(subscripts, con.lbounds()))
          idx.push_back(
              builder.createIntegerConstant(getLoc(), idxTy, dim - lb));
        array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                   charVal, idx);
      } while (con.IncrementSubscripts(subscripts));
      auto len = builder.createIntegerConstant(getLoc(), idxTy, con.LEN());
      return fir::CharArrayBoxValue{array, len, extents, lbounds};
    } else {
      // Convert Ev::ConstantSubs to SequenceType::Shape
      fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
      auto eleTy = converter.genType(TC, KIND);
      auto arrayTy = fir::SequenceType::get(shape, eleTy);
      mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
      Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
      bool foundRange = false;
      mlir::Value rangeValue;
      llvm::SmallVector<mlir::Value> rangeStartIdx;
      Fortran::evaluate::ConstantSubscripts rangeStartSubscripts;
      uint64_t elemsInRange = 0;
      const uint64_t minRangeSize = 2;

      do {
        auto constant =
            fir::getBase(genScalarLit<TC, KIND>(con.At(subscripts)));
        auto createIndexes = [&](Fortran::evaluate::ConstantSubscripts subs) {
          llvm::SmallVector<mlir::Value> idx;
          for (auto [dim, lb] : llvm::zip(subs, con.lbounds()))
            // Add normalized upper bound index to idx.
            idx.push_back(
                builder.createIntegerConstant(getLoc(), idxTy, dim - lb));

          return idx;
        };

        auto idx = createIndexes(subscripts);
        auto insVal = builder.createConvert(getLoc(), eleTy, constant);
        auto nextSubs = subscripts;

        // Check to see if the next value is the same as the current value
        bool nextIsSame = con.IncrementSubscripts(nextSubs) &&
                          con.At(subscripts) == con.At(nextSubs);
        bool newRange = (nextIsSame != foundRange) && !foundRange;
        bool endOfRange = (nextIsSame != foundRange) && foundRange;
        bool continueRange = nextIsSame && foundRange;

        if (newRange) {
          // Mark the start of the range
          rangeStartIdx = idx;
          rangeStartSubscripts = subscripts;
          rangeValue = insVal;
          foundRange = true;
          elemsInRange = 1;
        } else if (endOfRange) {
          ++elemsInRange;
          if (elemsInRange >= minRangeSize) {
            // Zip together the upper and lower bounds of the range for each
            // index in the form [lb0, up0, lb1, up1, ... , lbn, upn] to pass
            // to the InserOnEangeOp.
            llvm::SmallVector<mlir::Value> zippedRange;
            for (size_t i = 0; i < idx.size(); ++i) {
              zippedRange.push_back(rangeStartIdx[i]);
              zippedRange.push_back(idx[i]);
            }
            array = builder.create<fir::InsertOnRangeOp>(
                getLoc(), arrayTy, array, rangeValue, zippedRange);
          } else {
            while (true) {
              idx = createIndexes(rangeStartSubscripts);
              array = builder.create<fir::InsertValueOp>(
                  getLoc(), arrayTy, array, rangeValue, idx);
              if (rangeStartSubscripts == subscripts)
                break;
              con.IncrementSubscripts(rangeStartSubscripts);
            }
          }
          foundRange = false;
        } else if (continueRange) {
          // Loop until the end of the range is found.
          ++elemsInRange;
          continue;
        } else /* no range */ {
          // If a range has not been found then insert the current value.
          array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                     insVal, idx);
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
    auto opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      return genScalarLit<KIND>(opt.value(), con.LEN());
    } else {
      return genScalarLit<TC, KIND>(opt.value());
    }
  }
  template <Fortran::common::TypeCategory TC>
  ExtValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeKind<TC>> &con) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      auto opt = (*con).ToInt64();
      auto type = getSomeKindInteger();
      auto attr = builder.getIntegerAttr(type, opt);
      auto res = builder.create<mlir::ConstantOp>(getLoc(), type, attr);
      return res.getResult();
    } else if constexpr (TC == Fortran::common::TypeCategory::Derived) {
      if (auto ctor = con.GetScalarValue())
        return genval(ctor.value());
      fir::emitFatalError(getLoc(),
                          "constant of derived type has no constructor");
    } else {
      fir::emitFatalError(getLoc(), "unhandled constant of unknown kind");
    }
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    fir::emitFatalError(getLoc(),
                        "array constructor: lowering should not reach here");
  }

  ExtValue gen(const Fortran::evaluate::ComplexPart &x) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto exv = gen(x.complex());
    auto base = fir::getBase(exv);
    Fortran::lower::ComplexExprHelper helper{builder, loc};
    auto eleTy =
        helper.getComplexPartType(fir::dyn_cast_ptrEleTy(base.getType()));
    auto offset = builder.createIntegerConstant(
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
              if (auto str = p->AsString())
                return Fortran::lower::createStringLiteral(builder, getLoc(),
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
    auto lower = genunbox(s.lower());
    bounds.push_back(lower);
    if (auto upperBound = s.upper()) {
      auto upper = genunbox(*upperBound);
      bounds.push_back(upper);
    }
    Fortran::lower::CharacterExprHelper charHelper{builder, getLoc()};
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

  ExtValue genComponent(const Fortran::evaluate::Subscript &subs) {
    if (auto *s = std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
            &subs.u))
      return genval(s->value());
    fir::emitFatalError(getLoc(), "unhandled subscript case");
  }

  ExtValue genval(const Fortran::evaluate::Subscript &subs) {
    if (auto *s = std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
            &subs.u))
      return {genval(s->value())};
    llvm_unreachable("unhandled subscript case");
  }

  ExtValue gen(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  ExtValue genval(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return genval(x); }, dref.u);
  }

  // Helper function to turn the left-recursive Component structure into a list.
  // Returns the object used as the base coordinate for the component chain.
  static Fortran::evaluate::DataRef const *
  reverseComponents(const Fortran::evaluate::Component &cmpt,
                    std::list<const Fortran::evaluate::Component *> &list) {
    list.push_front(&cmpt);
    return std::visit(Fortran::common::visitors{
                          [&](const Fortran::evaluate::Component &x) {
                            return reverseComponents(x, list);
                          },
                          [&](auto &) { return &cmpt.base(); },
                      },
                      cmpt.base().u);
  }

  // Return the coordinate of the component reference
  ExtValue gen(const Fortran::evaluate::Component &cmpt) {
    std::list<const Fortran::evaluate::Component *> list;
    auto *base = reverseComponents(cmpt, list);
    llvm::SmallVector<mlir::Value> coorArgs;
    auto obj = gen(*base);
    auto ty = fir::dyn_cast_ptrEleTy(fir::getBase(obj).getType());
    auto loc = getLoc();
    auto fldTy = fir::FieldType::get(&converter.getMLIRContext());
    // FIXME: need to thread the LEN type parameters here.
    for (auto *field : list) {
      auto recTy = ty.cast<fir::RecordType>();
      const auto *sym = &field->GetLastSymbol();
      auto name = toStringRef(sym->name());
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(
          loc, fldTy, name, recTy, fir::getTypeParams(obj)));
      ty = recTy.getType(name);
    }
    ty = builder.getRefType(ty);
    return componentToExtendedValue(builder, loc, obj,
                                    builder.create<fir::CoordinateOp>(
                                        loc, ty, fir::getBase(obj), coorArgs));
  }

  ExtValue genval(const Fortran::evaluate::Component &cmpt) {
    return genLoad(gen(cmpt));
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  mlir::Type genSubType(mlir::Type arrTy, unsigned dims) {
    auto unwrapTy = arrTy.cast<fir::ReferenceType>().getEleTy();
    auto seqTy = unwrapTy.cast<fir::SequenceType>();
    auto shape = seqTy.getShape();
    assert(shape.size() > 0 && "removing columns for sequence sans shape");
    assert(dims <= shape.size() && "removing more columns than exist");
    fir::SequenceType::Shape newBnds;
    // follow Fortran semantics and remove columns (from right)
    auto e = shape.size() - dims;
    for (decltype(e) i{0}; i < e; ++i)
      newBnds.push_back(shape[i]);
    if (!newBnds.empty())
      return fir::SequenceType::get(newBnds, seqTy.getEleTy());
    return seqTy.getEleTy();
  }

  // Generate the code for a Bound value.
  ExtValue genval(const Fortran::semantics::Bound &bound) {
    if (bound.isExplicit()) {
      auto sub = bound.GetExplicit();
      if (sub.has_value())
        return genval(*sub);
      return genIntegerConstant<8>(builder.getContext(), 1);
    }
    TODO(getLoc(), "non explicit semantics::Bound lowering");
  }

  ExtValue genArrayRefComponent(const Fortran::evaluate::ArrayRef &aref) {
    auto component = gen(aref.base().GetComponent());
    auto base = fir::getBase(component);
    llvm::SmallVector<mlir::Value> args;
    // FIXME: the lower bounds should be apply, and in general,
    // this coordinate op will only work if the extents are compile
    // time constants. Otherwise, the coordinateOp need to be collapsed.
    for (auto &subsc : aref.subscript())
      args.push_back(genunbox(subsc));
    auto ty = genSubType(base.getType(), args.size());
    ty = builder.getRefType(ty);
    auto addr = builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
    return arrayElementToExtendedValue(builder, getLoc(), component, addr);
  }

  static bool isSlice(const Fortran::evaluate::ArrayRef &aref) {
    for (auto &sub : aref.subscript())
      if (std::holds_alternative<Fortran::evaluate::Triplet>(sub.u))
        return true;
    return false;
  }

  ExtValue gen(const Fortran::lower::SymbolBox &si,
               const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = si.getAddr();
    auto arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto seqTy = builder.getRefType(builder.getVarLenSeqTy(eleTy));
    auto refTy = builder.getRefType(eleTy);
    auto base = builder.createConvert(loc, seqTy, addr);
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto getLB = [&](const auto &arr, unsigned dim) -> mlir::Value {
      return arr.getLBounds().empty() ? one : arr.getLBounds()[dim];
    };
    auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
      mlir::Value total = zero;
      assert(arr.getExtents().size() == aref.subscript().size());
      delta = builder.createConvert(loc, idxTy, delta);
      unsigned dim = 0;
      for (auto [ext, sub] : llvm::zip(arr.getExtents(), aref.subscript())) {
        auto subVal = genComponent(sub);
        assert(fir::isUnboxedValue(subVal));
        auto val = builder.createConvert(loc, idxTy, fir::getBase(subVal));
        auto lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
        auto diff = builder.create<mlir::SubIOp>(loc, val, lb);
        auto prod = builder.create<mlir::MulIOp>(loc, delta, diff);
        total = builder.create<mlir::AddIOp>(loc, prod, total);
        if (ext)
          delta = builder.create<mlir::MulIOp>(loc, delta, ext);
        ++dim;
      }
      return builder.create<fir::CoordinateOp>(
          loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
    };
    return si.match(
        [&](const Fortran::lower::SymbolBox::FullDim &arr) -> ExtValue {
          // FIXME: this check can be removed when slicing is implemented
          if (isSlice(aref))
            fir::emitFatalError(
                getLoc(),
                "slice should be handled in array expression context");
          return genFullDim(arr, one);
        },
        [&](const Fortran::lower::SymbolBox::CharFullDim &arr) -> ExtValue {
          auto delta = arr.getLen();
          // If the length is known in the type, fir.coordinate_of will
          // already take the length into account.
          if (Fortran::lower::CharacterExprHelper::hasConstantLengthInType(arr))
            delta = one;
          return fir::CharBoxValue(genFullDim(arr, delta), arr.getLen());
        },
        [&](const Fortran::lower::SymbolBox::Box &arr) -> ExtValue {
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

  ExtValue genArrayCoorOp(const ExtValue &exv,
                          const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = fir::getBase(exv);
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto refTy = builder.getRefType(eleTy);
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> arrayCoorArgs;
    // The ArrayRef is expected to be scalar here, arrays are handled in array
    // expression lowering. So no vector subscript or triplet is expected here.
    for (const auto &sub : aref.subscript()) {
      auto subVal = genComponent(sub);
      assert(fir::isUnboxedValue(subVal));
      arrayCoorArgs.push_back(
          builder.createConvert(loc, idxTy, fir::getBase(subVal)));
    }
    auto shape = builder.createShape(loc, exv);
    llvm::SmallVector<mlir::Value> lengthParams;
    exv.match(
        [&](const fir::CharArrayBoxValue &arr) {
          lengthParams.emplace_back(arr.getLen());
        },
        [&](const fir::BoxValue &arr) {
          auto lengths = arr.getExplicitParameters();
          lengthParams.append(lengths.begin(), lengths.end());
        },
        [](const auto &) {});
    auto elementAddr = builder.create<fir::ArrayCoorOp>(
        loc, refTy, addr, shape, /*slice=*/mlir::Value{}, arrayCoorArgs,
        lengthParams);
    return arrayElementToExtendedValue(builder, loc, exv, elementAddr);
  }

  // Return the coordinate of the array reference
  ExtValue gen(const Fortran::evaluate::ArrayRef &aref) {
    if (aref.base().IsSymbol()) {
      auto &symbol = aref.base().GetFirstSymbol();
      if (generateArrayCoordinate)
        return genArrayCoorOp(gen(symbol), aref);
      auto si = symMap.lookupSymbol(symbol);
      si = si.match(
          [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &x)
              -> Fortran::lower::SymbolBox {
            return genAllocatableOrPointerUnbox(x);
          },
          [](const auto &x) -> Fortran::lower::SymbolBox { return x; });
      auto isIrBox = si.getAddr().getType().isa<fir::BoxType>();
      if (!si.hasConstantShape() && !isIrBox)
        return gen(si, aref);
      auto box = gen(symbol);
      auto base = fir::getBase(box);
      unsigned i = 0;
      llvm::SmallVector<mlir::Value> args;
      auto loc = getLoc();
      for (auto &subsc : aref.subscript()) {
        auto subVal = genComponent(subsc);
        assert(fir::isUnboxedValue(subVal));
        auto val = fir::getBase(subVal);
        auto ty = val.getType();
        auto adj = getLBound(si, i++, ty);
        assert(adj && "boxed value not handled");
        args.push_back(builder.create<mlir::SubIOp>(loc, ty, val, adj));
      }

      auto seqTy =
          fir::dyn_cast_ptrOrBoxEleTy(base.getType()).cast<fir::SequenceType>();
      assert(args.size() == seqTy.getDimension());
      auto ty = builder.getRefType(seqTy.getEleTy());
      auto addr = builder.create<fir::CoordinateOp>(loc, ty, base, args);
      return arrayElementToExtendedValue(builder, loc, box, addr);
    }
    return genArrayRefComponent(aref);
  }

  mlir::Value getLBound(const Fortran::lower::SymbolBox &box, unsigned dim,
                        mlir::Type ty) {
    assert(box.hasRank());
    if (box.hasSimpleLBounds())
      return builder.createIntegerConstant(getLoc(), ty, 1);
    return builder.createConvert(getLoc(), ty, box.getLBound(dim));
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
    return converter.genType(dt.GetDerivedTypeSpec());
  }

  /// Apply the function `func` and return a reference to the resultant value.
  /// This is required for lowering expressions such as `f1(f2(v))`.
  template <typename A>
  ExtValue gen(const Fortran::evaluate::FunctionRef<A> &func) {
    if (!func.GetType().has_value())
      mlir::emitError(getLoc(), "internal: a function must have a type");
    auto resTy = genType(*func.GetType());
    auto retVal = genProcedureRef(func, {resTy});
    auto retValBase = fir::getBase(retVal);
    if (fir::conformsWithPassByRef(retValBase.getType()))
      return retVal;
    auto mem = builder.create<fir::AllocaOp>(getLoc(), retValBase.getType());
    builder.create<fir::StoreOp>(getLoc(), retValBase, mem);
    return fir::substBase(retVal, mem.getResult());
  }

  /// Helper to lower intrinsic arguments for inquiry intrinsic.
  ExtValue
  lowerIntrinsicArgumentAsInquired(const Fortran::lower::SomeExpr &expr) {
    const auto *sym = Fortran::evaluate::UnwrapWholeSymbolDataRef(expr);
    if (sym && Fortran::semantics::IsAllocatableOrPointer(*sym))
      return genMutableBoxValue(expr);
    return gen(expr);
  }

  /// Generate a call to an intrinsic function.
  ExtValue
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  llvm::Optional<mlir::Type> resultType) {
    llvm::SmallVector<ExtValue> operands;

    llvm::StringRef name = intrinsic.name;
    const auto *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg);
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
      auto lowerAs = Fortran::lower::lowerIntrinsicArgumentAs(
          getLoc(), *argLowering, dummy.name);
      switch (lowerAs) {
      case Fortran::lower::LowerIntrinsicArgAs::Value:
        operands.emplace_back(genval(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Addr:
        operands.emplace_back(gen(*expr));
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
  bool isCharacterType(const A &exp) {
    if (auto type = exp.GetType())
      return type->category() == Fortran::common::TypeCategory::Character;
    return false;
  }

  /// helper to detect statement functions
  static bool
  isStatementFunctionCall(const Fortran::evaluate::ProcedureRef &procRef) {
    if (const auto *symbol = procRef.proc().GetSymbol())
      if (const auto *details =
              symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
        return details->stmtFunction().has_value();
    return false;
  }
  /// Generate Statement function calls
  ExtValue genStmtFunctionRef(const Fortran::evaluate::ProcedureRef &procRef) {
    const auto *symbol = procRef.proc().GetSymbol();
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
      // scalars, so just pass the box with the address.
      symMap.addSymbol(*arg, gen(*expr));
    }

    // Explicitly map statement function host associated symbols to their
    // parent scope lowered symbol box.
    for (const Fortran::semantics::SymbolRef &sym :
         Fortran::evaluate::CollectSymbols(*details.stmtFunction())) {
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
        if (!symMap.lookupSymbol(*sym)) {
          symMap.addSymbol(
              *sym, symMap.lookupSymbol(details->symbol()).toExtendedValue());
        }
      }
    }

    auto result = genval(details.stmtFunction().value());
    LLVM_DEBUG(llvm::dbgs() << "stmt-function: " << result << '\n');
    symMap.popScope();
    return result;
  }

  /// Helper to package a value and its properties into an ExtendedValue.
  ExtValue toExtendedValue(mlir::Value base,
                           llvm::ArrayRef<mlir::Value> extents,
                           llvm::ArrayRef<mlir::Value> lengths) {
    auto type = base.getType();
    if (type.isa<fir::BoxType>())
      return fir::BoxValue(base, /*lbounds=*/{}, lengths, extents);
    if (auto pointedType = fir::dyn_cast_ptrEleTy(type))
      type = pointedType;
    if (type.isa<fir::BoxType>())
      return fir::MutableBoxValue(base, lengths, /*mutableProperties*/ {});
    if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
      if (seqTy.getDimension() != extents.size())
        fir::emitFatalError(getLoc(), "incorrect number of extents for array");
      if (seqTy.getEleTy().isa<fir::CharacterType>()) {
        if (lengths.empty())
          fir::emitFatalError(getLoc(), "missing length for character");
        assert(lengths.size() == 1);
        return fir::CharArrayBoxValue(base, lengths[0], extents);
      }
      return fir::ArrayBoxValue(base, extents);
    }
    if (type.isa<fir::CharacterType>()) {
      if (lengths.empty())
        fir::emitFatalError(getLoc(), "missing length for character");
      assert(lengths.size() == 1);
      return fir::CharBoxValue(base, lengths[0]);
    }
    return base;
  }

  ExtValue genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                           llvm::Optional<mlir::Type> resultType) {
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType);

    if (isStatementFunctionCall(procRef))
      return genStmtFunctionRef(procRef);

    auto loc = getLoc();
    Fortran::lower::CallerInterface caller(procRef, converter);
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    llvm::SmallVector<fir::MutableBoxValue> mutableModifiedByCall;

    auto callSiteType = caller.genFunctionType();
    for (const auto &arg : caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      auto argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        caller.placeInput(arg, builder.create<fir::AbsentOp>(loc, argTy));
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

      if (arg.passBy == PassBy::Value) {
        auto argVal = genval(*expr);
        if (!fir::isUnboxedValue(argVal))
          fir::emitFatalError(
              loc, "internal error: passing non trivial value by value");
        caller.placeInput(arg, fir::getBase(argVal));
        continue;
      }

      if (arg.passBy == PassBy::MutableBox) {
        auto mutableBox = genMutableBoxValue(*expr);
        auto irBox = Fortran::lower::getMutableIRBox(builder, loc, mutableBox);
        caller.placeInput(arg, irBox);
        // TODO: no need to add this to the list if intent(in)
        mutableModifiedByCall.emplace_back(std::move(mutableBox));
        continue;
      }

      auto argRef = genExtArg(*expr, arg.passBy == PassBy::Box);

      auto helper = Fortran::lower::CharacterExprHelper{builder, loc};
      if (arg.passBy == PassBy::BaseAddress) {
        auto baseAddr = fir::getBase(argRef);
        if (baseAddr.getType().isa<fir::BoxType>())
          TODO(loc, "copy-in copy-out around F77 calls");
        caller.placeInput(arg, baseAddr);
      } else if (arg.passBy == PassBy::BoxChar) {
        auto boxChar = argRef.match(
            [&](const fir::CharBoxValue &x) { return helper.createEmbox(x); },
            [&](const fir::CharArrayBoxValue &x) {
              return helper.createEmbox(x);
            },
            [&](const fir::BoxValue &x) -> mlir::Value {
              // Beware, descriptor content might have to be copied before
              // and after the call to a contiguous character argument.
              TODO(loc, "lowering actual arguments descriptor to boxchar");
            },
            [&](const auto &) -> mlir::Value {
              fir::emitFatalError(
                  loc, "internal error: actual argument is not a character");
            });
        caller.placeInput(arg, boxChar);
      } else if (arg.passBy == PassBy::Box) {
        auto box = builder.createBox(loc, argRef);
        // Before lowering to an address, handle the allocatable/pointer actual
        // argument to optional fir.box dummy. It is legal to pass
        // unallocated/disassociated entity to an optional. In this case, an
        // absent fir.box must be created instead of a fir.box with a null value
        // (Fortran 2018 15.5.2.12 point 1).
        const auto *sym = Fortran::evaluate::UnwrapWholeSymbolDataRef(*expr);
        if (arg.isOptional && sym &&
            Fortran::semantics::IsAllocatableOrPointer(*sym)) {
          // Note that passing an absent allocatable to a non-allocatable
          // optional dummy argument is illegal (15.5.2.12 point 3 (8)). So
          // nothing has to be done to generate an absent argument in this case,
          // and it is OK to unconditionally read the mutable box here.
          auto mutableBox = genMutableBoxValue(*expr);
          auto isAllocated = Fortran::lower::genIsAllocatedOrAssociatedTest(
              builder, loc, mutableBox);
          auto absent = builder.create<fir::AbsentOp>(loc, argTy);
          // Need the box types to be exactly similar for the selectOp.
          auto convertedBox = builder.createConvert(loc, argTy, box);
          caller.placeInput(arg, builder.create<mlir::SelectOp>(
                                     loc, isAllocated, convertedBox, absent));
        } else {
          caller.placeInput(arg, box);
        }
      } else if (arg.passBy == PassBy::AddressAndLength) {
        caller.placeAddressAndLengthInput(arg, fir::getBase(argRef),
                                          fir::getLen(argRef));
      } else {
        llvm_unreachable("pass by value not handled here");
      }
    }

    // Handle cases where caller must allocate the result or a fir.box for it.
    bool mustPopSymMap = false;
    if (caller.mustMapInterfaceSymbols()) {
      symMap.pushScope();
      mustPopSymMap = true;
      Fortran::lower::mapCallInterfaceSymbols(converter, caller, symMap);
    }
    auto idxTy = builder.getIndexType();
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
      auto type = caller.getResultStorageType();
      if (type.isa<fir::SequenceType>())
        caller.walkResultExtents([&](const Fortran::lower::SomeExpr &e) {
          extents.emplace_back(lowerSpecExpr(e));
        });
      caller.walkResultLengths([&](const Fortran::lower::SomeExpr &e) {
        lengths.emplace_back(lowerSpecExpr(e));
      });
      /// Result lengths parameters should not be provided to box storage
      /// allocation and save_results, but they are still useful information to
      /// keep in the ExtentdedValue if non-deferred.
      if (!type.isa<fir::BoxType>())
        resultLengths = lengths;
      auto temp =
          builder.createTemporary(loc, type, ".result", extents, resultLengths);
      return toExtendedValue(temp, extents, lengths);
    }();

    if (mustPopSymMap)
      symMap.popScope();

    // Place allocated result or prepare the fir.save_result arguments.
    mlir::Value arrayResultShape;
    if (allocatedResult) {
      if (auto resultArg = caller.getPassedResult()) {
        if (resultArg->passBy == PassBy::AddressAndLength) {
          caller.placeAddressAndLengthInput(*resultArg,
                                            fir::getBase(*allocatedResult),
                                            fir::getLen(*allocatedResult));
        } else if (resultArg->passBy == PassBy::BaseAddress) {
          caller.placeInput(*resultArg, fir::getBase(*allocatedResult));
        } else {
          fir::emitFatalError(
              loc, "only expect character scalar result to be passed by ref");
        }
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
    // Casts are inserted as needed below to acomodate this.

    // The mlir::FuncOp type prevails, unless it has a different number of
    // arguments which can happen in legal program if it was passed as a dummy
    // procedure argument earlier with no further type information.
    mlir::Value funcPointer;
    mlir::SymbolRefAttr funcSymbolAttr;
    if (const auto *sym = caller.getIfIndirectCallSymbol()) {
      funcPointer = symMap.lookupSymbol(*sym).getAddr();
      assert(funcPointer &&
             "dummy procedure or procedure pointer not in symbol map");
    } else {
      auto funcOpType = caller.getFuncOp().getType();
      // Deal with argument number mismatch by making a function pointer so that
      // function type cast can be inserted.
      auto symbolAttr = builder.getSymbolRefAttr(caller.getMangledName());
      if (callSiteType.getNumResults() != funcOpType.getNumResults() ||
          callSiteType.getNumInputs() != funcOpType.getNumInputs()) {
        // Do not emit a warning here because this can happen in legal program
        // if the function is not defined here and it was first passed as an
        // argument without any more information.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else if (callSiteType.getResults() != funcOpType.getResults()) {
        // Implicit interface result type mismatch are not standard Fortran, but
        // some compilers are not complaining about it.  The front end is not
        // protecting lowering from this currently. Support this with a
        // discouraging warning.
        mlir::emitWarning(loc,
                          "return type mismatches were never standard"
                          " compliant and may lead to undefined behavior.");
        // Cast the actual function to the current caller implicit type because
        // that is the behavior we would get if we could not see the definition.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else {
        funcSymbolAttr = symbolAttr;
      }
    }
    auto funcType = funcPointer ? callSiteType : caller.getFuncOp().getType();
    llvm::SmallVector<mlir::Value> operands;
    // First operand of indirect call is the function pointer. Cast it to
    // required function type for the call to handle procedures that have a
    // compatible interface in Fortran, but that have different signatures in
    // FIR.
    if (funcPointer)
      operands.push_back(builder.createConvert(loc, funcType, funcPointer));

    // Deal with potential mismatches in arguments types. Passing an array to a
    // scalar argument should for instance be tolerated here.
    for (auto [fst, snd] :
         llvm::zip(caller.getInputs(), funcType.getInputs())) {
      auto cast = builder.convertWithSemantics(getLoc(), snd, fst);
      operands.push_back(cast);
    }

    auto call = builder.create<fir::CallOp>(loc, funcType.getResults(),
                                            funcSymbolAttr, operands);

    if (caller.mustSaveResult())
      builder.create<fir::SaveResultOp>(
          loc, call.getResult(0), fir::getBase(allocatedResult.getValue()),
          arrayResultShape, resultLengths);

    // Sync pointers and allocatables that may have been modified during the
    // call.
    for (const auto &mutableBox : mutableModifiedByCall)
      Fortran::lower::syncMutableBoxFromIRBox(builder, loc, mutableBox);
    // Handle case where result was passed as argument

    if (allocatedResult) {
      return allocatedResult->match(
          [&](const fir::MutableBoxValue &box) {
            if (box.isAllocatable()) {
              // 9.7.3.2 point 4. Finalize allocatables.
              auto *bldr = &converter.getFirOpBuilder();
              stmtCtx.attachCleanup(
                  [=]() { Fortran::lower::genFinalization(*bldr, loc, box); });
            }
            // Pointers and allocatable do not appear as allocatable/pointer
            // variable on the caller side (see 8.5.3 note 1 for allocatables).
            return Fortran::lower::genMutableBoxRead(builder, loc, box)
                .toExtendedValue();
          },
          [&](const auto &) { return *allocatedResult; });
    }

    if (!resultType.hasValue())
      return mlir::Value{}; // subroutine call
    // For now, Fortran return values are implemented with a single MLIR
    // function return value.
    assert(call.getNumResults() == 1 &&
           "Expected exactly one result in FUNCTION call");
    return call.getResult(0);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
             &funRef) {
    auto retTy = converter.genType(TC, KIND);
    return genProcedureRef(funRef, {retTy});
  }

  ExtValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    llvm::Optional<mlir::Type> resTy;
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
    auto expr = toEvExpr(x);
    auto optShape =
        Fortran::evaluate::GetShape(converter.getFoldingContext(), expr);
    return Fortran::lower::createSomeArrayTempValue(converter, optShape, expr,
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
    return Fortran::lower::createSomeArrayBox(converter, toEvExpr(x), symMap,
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
    // Whole array symbols and results of transformational functions already
    // have a storage and the scalar expression lowering path is used to not
    // create a new temporary storage.
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x) ||
        isTransformationalRef(x))
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
    auto exv = genval(a);
    auto valBase = fir::getBase(exv);
    // Functions are always referent.
    if (valBase.getType().template isa<mlir::FunctionType>() ||
        fir::conformsWithPassByRef(valBase.getType()))
      return exv;

    // Since `a` is not itself a valid referent, determine its value and
    // create a temporary location at the begining of the function for
    // referencing.
    auto func = builder.getFunction();
    auto initPos = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&func.front());
    auto mem = builder.create<fir::AllocaOp>(getLoc(), valBase.getType());
    builder.restoreInsertionPoint(initPos);
    builder.create<fir::StoreOp>(getLoc(), valBase, mem);
    return fir::substBase(exv, mem.getResult());
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
  Fortran::lower::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  bool inInitializer;
  bool useBoxArg{false}; // expression lowered as argument
};
} // namespace

//===----------------------------------------------------------------------===//
//
// Lowering of array expressions.
//
//===----------------------------------------------------------------------===//

// Helper for changing the semantics in a given context. Preserves the current
// semantics which is resumed when the "push" goes out of scope.
#define PushSemantics(PushVal)                                                 \
  [[maybe_unused]] auto pushSemanticsLocalVariable97201 =                      \
      Fortran::common::ScopedSet(semant, PushVal);

namespace {
class ArrayExprLowering {
  struct IterationSpace {
    IterationSpace() = default;

    template <typename A>
    explicit IterationSpace(mlir::Value inArg, mlir::Value outRes,
                            llvm::iterator_range<A> range)
        : inArg{inArg}, outRes{outRes}, indices{range.begin(), range.end()} {}

    mlir::Value innerArgument() const { return inArg; }
    mlir::Value outerResult() const { return outRes; }
    llvm::ArrayRef<mlir::Value> iterVec() const { return indices; }

    /// Set (rewrite) the Value at a given index.
    void setIndexValue(std::size_t i, mlir::Value v) {
      assert(i < indices.size());
      indices[i] = v;
    }

    void insertIndexValue(std::size_t i, mlir::Value v) {
      assert(i <= indices.size());
      indices.insert(indices.begin() + i, v);
    }

    void setElement(mlir::Value ele) {
      assert(!element);
      element = ele;
    }

    mlir::Value getElement() const {
      assert(element);
      return element;
    }

  private:
    mlir::Value inArg;
    mlir::Value outRes;
    mlir::Value element;
    llvm::SmallVector<mlir::Value> indices;
  };

public:
  using ExtValue = fir::ExtendedValue;
  using IterSpace = const IterationSpace &;      // active iteration space
  using CC = std::function<ExtValue(IterSpace)>; // current continuation
  using PC =
      std::function<IterationSpace(IterSpace)>; // projection continuation

  static void lowerArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut};
    ael.lowerArrayAssignment(lhs, rhs);
  }

  void lowerArrayAssignment(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs) {
    auto loc = getLoc();
    destShape = Fortran::evaluate::GetShape(converter.getFoldingContext(), lhs);
    lowerArrayProjectedSubspace(lhs);
    semant = ConstituentSemantics::RefTransparent;
    auto exv = lowerArrayExpression(rhs);
    builder.create<fir::ArrayMergeStoreOp>(loc, destination, fir::getBase(exv),
                                           destination.memref(),
                                           destination.typeparams());
  }

  /// Entry point for masked array assignment, Fortran's WHERE. This has the
  /// same semantics as ordinary array assignment except that the RHS is a
  /// projected array value based on the mask condition(s).
  static void lowerMaskedArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
      Fortran::lower::MaskExpr &masks) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &masks};
    ael.lowerArrayAssignment(lhs, rhs);
  }

  /// Entry point for when an array expression appears on the lhs of an
  /// assignment. In the default case, the rhs is fully evaluated prior to any
  /// of the results being written back to the lhs. (CopyInCopyOut semantics.)
  static fir::ArrayLoadOp lowerArraySubspace(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut};
    return ael.lowerArraySubspace(expr);
  }

  fir::ArrayLoadOp lowerArraySubspace(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          auto exv = f(IterationSpace{});
          if (auto *defOp = fir::getBase(exv).getDefiningOp())
            if (auto arrLd = mlir::dyn_cast<fir::ArrayLoadOp>(defOp))
              return arrLd;
          fir::emitFatalError(getLoc(), "array must be loaded");
        },
        exp.u);
  }

  /// Here the target subspace is not necessarily contiguous. The ArrayUpdate
  /// continuation is implicitly returned in `ccDest` and the ArrayLoad in
  /// `destination`.
  void lowerArrayProjectedSubspace(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    return std::visit([&](const auto &e) { ccDest = genarr(e); }, exp.u);
  }

  /// Entry point for when an array expression appears in a context where the
  /// result must be boxed. (BoxValue semantics.)
  static ExtValue lowerArrayExpressionBoxed(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::BoxValue};
    return ael.lowerArrayExprBoxed(expr);
  }

  ExtValue lowerArrayExprBoxed(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          auto exv = f(IterationSpace{});
          if (auto *defOp = fir::getBase(exv).getDefiningOp();
              mlir::isa<fir::EmboxOp>(defOp))
            return exv;
          fir::emitFatalError(getLoc(), "array must be emboxed");
        },
        exp.u);
  }

  /// Entry point into lowering an expression with rank. This entry point is for
  /// lowering a rhs expression, for example. (RefTransparent semantics.)
  static ExtValue lowerSomeNewArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const std::optional<Fortran::evaluate::Shape> &shape,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {

    ArrayExprLowering ael{converter, stmtCtx, symMap, /*destination=*/{},
                          shape};
    auto loopRes = ael.lowerArrayExpression(expr);
    auto dest = ael.destination;
    auto tempRes = dest.memref();
    auto &builder = converter.getFirOpBuilder();
    auto loc = converter.getCurrentLocation();
    builder.create<fir::ArrayMergeStoreOp>(loc, dest, fir::getBase(loopRes),
                                           tempRes, dest.typeparams());

    auto arrTy =
        fir::dyn_cast_ptrEleTy(tempRes.getType()).cast<fir::SequenceType>();
    if (auto charTy =
            arrTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      if (charTy.getLen() <= 0)
        TODO(loc, "CHARACTER does not have constant LEN");
      auto len = builder.createIntegerConstant(
          loc, builder.getCharacterLengthType(), charTy.getLen());
      return fir::CharArrayBoxValue(tempRes, len, dest.getExtents());
    }
    return fir::ArrayBoxValue(tempRes, dest.getExtents());
  }

  /// CHARACTER and derived type elements are treated as memory references. The
  /// numeric types are treated as values.
  static bool isAdjustedArrayElementType(mlir::Type t) {
    return fir::isa_char(t) || t.isa<fir::RecordType>();
  }
  static bool elementTypeWasAdjusted(mlir::Type t) {
    if (auto ty = t.dyn_cast<fir::ReferenceType>())
      return isAdjustedArrayElementType(ty.getEleTy());
    return false;
  }
  static mlir::Type adjustedArrayElementType(mlir::Type t) {
    return isAdjustedArrayElementType(t) ? fir::ReferenceType::get(t) : t;
  }

  /// For an elemental array expression.
  /// 1. Lower the scalars and array loads.
  /// 2. Create the iteration space.
  /// 3. Create the element-by-element computation in the loop.
  /// 4. Return the resulting array value.
  ExtValue lowerArrayExpression(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    auto resultType = converter.genType(exp);
    return std::visit(
        [&](const auto &e) {
          auto loc = getLoc();
          auto f = genarr(e);
          auto [iterSpace, insPt] = genIterSpace(resultType);
          auto innerArg = iterSpace.innerArgument();
          // Convert to array elemental type is needed for logical.
          auto eleTy = innerArg.getType().cast<fir::SequenceType>().getEleTy();
          auto exv = f(iterSpace);
          auto element = exv.match(
              [&](const fir::UnboxedValue &v) {
                return builder.createConvert(loc, eleTy, v);
              },
              [&](const fir::CharBoxValue &v) {
                return builder.createConvert(loc, builder.getRefType(eleTy),
                                             v.getBuffer());
              },
              [&](const auto &) -> mlir::Value {
                fir::emitFatalError(loc, "unhandled value for array update");
              });
          iterSpace.setElement(element);
          auto resTy = adjustedArrayElementType(innerArg.getType());
          mlir::Value upd =
              ccDest.hasValue()
                  ? fir::getBase(ccDest.getValue()(iterSpace))
                  : builder.create<fir::ArrayUpdateOp>(
                        loc, resTy, innerArg, element, iterSpace.iterVec(),
                        destination.typeparams());
          builder.create<fir::ResultOp>(loc, upd);
          builder.restoreInsertionPoint(insPt);
          return fir::substBase(exv, iterSpace.outerResult());
        },
        exp.u);
  }

  /// Compute the shape of a slice.
  llvm::SmallVector<mlir::Value> computeSliceShape(mlir::Value slice) {
    llvm::SmallVector<mlir::Value> slicedShape;
    auto slOp = mlir::cast<fir::SliceOp>(slice.getDefiningOp());
    auto triples = slOp.triples();
    auto idxTy = builder.getIndexType();
    auto loc = getLoc();
    for (unsigned i = 0, end = triples.size(); i < end; i += 3) {
      if (!mlir::isa_and_nonnull<fir::UndefOp>(
              triples[i + 1].getDefiningOp())) {
        // (..., lb:ub:step, ...) case:  extent = (ub-lb+step)/step
        auto lb = builder.createConvert(loc, idxTy, triples[i]);
        auto ub = builder.createConvert(loc, idxTy, triples[i + 1]);
        auto step = builder.createConvert(loc, idxTy, triples[i + 2]);
        auto diff = builder.create<mlir::SubIOp>(loc, ub, lb);
        auto add = builder.create<mlir::AddIOp>(loc, diff, step);
        slicedShape.emplace_back(
            builder.create<mlir::SignedDivIOp>(loc, add, step));
      }
      // else (..., i, ...) case: dimension is dropped (do nothing).
    }
    return slicedShape;
  }

  /// Get the shape from an array load.
  llvm::SmallVector<mlir::Value> getShape(fir::ArrayLoadOp arrayLoad) {
    if (arrayLoad.slice())
      return computeSliceShape(arrayLoad.slice());
    auto memref = arrayLoad.memref();
    if (memref.getType().isa<fir::BoxType>())
      return Fortran::lower::readExtents(builder, getLoc(),
                                         fir::BoxValue{memref});
    auto extents = arrayLoad.getExtents();
    return {extents.begin(), extents.end()};
  }

  /// Generate the shape of the array expressions based on the destination and
  /// operand array loads, or the optional evaluate::Shape.
  llvm::SmallVector<mlir::Value> genIterationShape() {
    // Use the optional evaluate::Shape if it has constant expressions,
    // otherwise, use the runtime destination or operand shapes. This could be
    // finned to find the shape that is the most expressive (for instance, the
    // one with the most constant extents).
    auto loc = getLoc();
    auto destShapeIsConstant = [&](const auto &shape) -> bool {
      for (const auto &s : shape)
        if (!s || !Fortran::evaluate::IsConstantExpr(*s))
          return false;
      return true;
    };
    if (destShape.has_value() && destShapeIsConstant(*destShape)) {
      auto idxTy = builder.getIndexType();
      llvm::SmallVector<mlir::Value> shape;
      for (const auto &s : *destShape)
        shape.emplace_back(builder.createConvert(
            loc, idxTy, convertOptExtentExpr(converter, stmtCtx, s)));
      return shape;
    }
    if (destination)
      return getShape(destination);
    if (!arrayOperandLoads.empty())
      return getShape(arrayOperandLoads[0]);
    fir::emitFatalError(loc, "failed to compute the array expression shape");
  }

  /// Build the iteration space into which the array expression will be lowered.
  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint>
  genIterSpace(mlir::Type resultType) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    llvm::SmallVector<mlir::Value> loopUppers;

    // Lower the mask expressions, if any.
    if (masks && !masks->empty()) {
      // Mask expressions are array expressions too.
      for (const auto *e : masks->getExprs())
        if (e && !masks->vmap.count(e)) {
          ScalarExprLowering sel(getLoc(), converter, symMap, stmtCtx);
          auto tmp = fir::getBase(sel.asArray(*e));
          masks->vmap.try_emplace(e, tmp);
        }
    }

    auto shape = genIterationShape();
    // Allocate a storage for the result is it is not already provided.
    if (!destination)
      destination = createAndLoadSomeArrayTemp(resultType, shape);
    // Convert the shape to closed interval form.
    for (auto extent : shape) {
      auto ub = builder.createConvert(loc, idxTy, extent);
      auto up = builder.create<mlir::SubIOp>(loc, ub, one);
      loopUppers.push_back(up);
    }

    // Iteration space is created with outermost columns, innermost rows
    fir::DoLoopOp inner;
    auto innerArg = destination.getResult();
    mlir::Value outerRes;
    const auto loopDepth = loopUppers.size();
    llvm::SmallVector<mlir::Value> ivars;
    auto insPt = builder.saveInsertionPoint();
    assert(loopDepth > 0);
    for (auto i : llvm::enumerate(llvm::reverse(loopUppers))) {
      if (i.index() > 0) {
        assert(inner);
        builder.setInsertionPointToStart(inner.getBody());
      }
      auto loop = builder.create<fir::DoLoopOp>(
          loc, zero, i.value(), one, /*unordered=*/false,
          /*finalCount=*/false, mlir::ValueRange{innerArg});
      innerArg = loop.getRegionIterArgs().front();
      // Store induction vars in column major order, which is what FIR array ops
      // expect.
      ivars.push_back(loop.getInductionVar());
      if (!outerRes)
        outerRes = loop.getResult(0);
      if (!inner)
        insPt = builder.saveInsertionPoint();
      if (i.index() < loopDepth - 1) {
        // Add the fir.result for all loops except the innermost one.
        builder.setInsertionPointToStart(loop.getBody());
        builder.create<fir::ResultOp>(loc, innerArg);
      }
      inner = loop;
    }

    // move insertion point inside loop nest
    builder.setInsertionPointToStart(inner.getBody());

    // put loop variables in row to column order
    IterationSpace iters{innerArg, outerRes, llvm::reverse(ivars)};

    // Generate the mask conditional structure, if there are masks.
    if (masks && !masks->empty()) {
      auto genCond = [&](mlir::Value tmp, IterSpace iters) {
        auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(tmp.getType());
        auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
        auto eleRefTy = builder.getRefType(eleTy);
        auto i1Ty = builder.getI1Type();
        auto addr = builder.create<fir::CoordinateOp>(loc, eleRefTy, tmp,
                                                      iters.iterVec());
        auto load = builder.create<fir::LoadOp>(loc, addr);
        return builder.createConvert(loc, i1Ty, load);
      };

      // Handle the negated conditions. See 10.2.3.2p4 as to why this control
      // structure is produced.
      auto maskExprs = masks->getExprs();
      auto size = maskExprs.size() - 1;
      for (decltype(size) i = 0; i < size; ++i) {
        auto ifOp = builder.create<fir::IfOp>(
            loc, mlir::TypeRange{innerArg.getType()},
            fir::getBase(genCond(masks->vmap.lookup(maskExprs[i]), iters)),
            /*withElseRegion=*/true);
        builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        builder.create<fir::ResultOp>(loc, innerArg);
        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
      }

      // The last condition is either non-negated or unconditionally negated.
      if (maskExprs[size]) {
        auto ifOp = builder.create<fir::IfOp>(
            loc, mlir::TypeRange{innerArg.getType()},
            fir::getBase(genCond(masks->vmap.lookup(maskExprs[size]), iters)),
            /*withElseRegion=*/true);
        builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        builder.create<fir::ResultOp>(loc, innerArg);
        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
      } else {
        // do nothing
      }
    }

    // We're ready to lower the body of this loop nest now.
    return {iters, insPt};
  }

  fir::ArrayLoadOp
  createAndLoadSomeArrayTemp(mlir::Type type,
                             llvm::ArrayRef<mlir::Value> shape) {
    auto seqTy = type.dyn_cast<fir::SequenceType>();
    assert(seqTy && "must be an array");
    auto loc = getLoc();
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
    auto *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> idxShape;
    for (auto s : shape)
      idxShape.push_back(builder.createConvert(loc, idxTy, s));
    auto shapeTy = fir::ShapeType::get(builder.getContext(), idxShape.size());
    auto shapeOp = builder.create<fir::ShapeOp>(loc, shapeTy, idxShape);
    mlir::Value slice;
    return builder.create<fir::ArrayLoadOp>(loc, seqTy, temp, shapeOp, slice,
                                            llvm::None);
  }

  //===--------------------------------------------------------------------===//
  // Expression traversal and lowering.
  //===--------------------------------------------------------------------===//

  // Lower the expression in a scalar context.
  template <typename A>
  ExtValue asScalar(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.genval(x);
  }

  // Lower the expression in a scalar context to a (boxed) reference.
  template <typename A>
  ExtValue asScalarRef(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.gen(x);
  }

  // An expression with non-zero rank is an array expression.
  template <typename A>
  static bool isArray(const A &x) {
    return x.Rank() != 0;
  }

  // Attribute for an alloca that is a trivial adaptor for converting a value to
  // pass-by-ref semantics for a VALUE parameter. The optimizer may be able to
  // eliminate these.
  mlir::NamedAttribute getAdaptToByRefAttr() {
    return {mlir::Identifier::get("adapt.valuebyref", builder.getContext()),
            builder.getUnitAttr()};
  }

  // A procedure reference to a Fortran elemental intrinsic procedure.
  CC genElementalIntrinsicProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      llvm::Optional<mlir::Type> retTy,
      const Fortran::evaluate::SpecificIntrinsic &intrinsic) {
    llvm::SmallVector<CC> operands;
    llvm::StringRef name = intrinsic.name;
    const auto *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    auto loc = getLoc();
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back([=](IterSpace) { return mlir::Value{}; });
      } else if (!argLowering) {
        // No argument lowering instruction, lower by value.
        PushSemantics(ConstituentSemantics::RefTransparent);
        auto lambda = genarr(*expr);
        operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
      } else {
        // Ad-hoc argument lowering handling.
        switch (Fortran::lower::lowerIntrinsicArgumentAs(getLoc(), *argLowering,
                                                         dummy.name)) {
        case Fortran::lower::LowerIntrinsicArgAs::Value: {
          PushSemantics(ConstituentSemantics::RefTransparent);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Addr: {
          // Note: assume does not have Fortran VALUE attribute semantics.
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
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
                                              stmtCtx);
    };
  }

  // A procedure reference to a user-defined elemental procedure.
  CC genElementalUserDefinedProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      llvm::Optional<mlir::Type> retTy) {
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    Fortran::lower::CallerInterface caller(procRef, converter);
    llvm::SmallVector<CC> operands(caller.getNumFIRArguments());
    auto loc = getLoc();
    auto callSiteType = caller.genFunctionType();
    for (const auto &arg : caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      auto argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        auto absent = builder.create<fir::AbsentOp>(loc, argTy);
        operands[arg.firArgument] = [=](IterSpace) { return absent; };
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

      LLVM_DEBUG(expr->AsFortran(llvm::dbgs()
                                 << "argument: " << arg.firArgument << " = [")
                 << "]\n");
      switch (arg.passBy) {
      case PassBy::Value: {
        // True pass-by-value semantics.
        PushSemantics(ConstituentSemantics::RefTransparent);
        auto lambda = genarr(*expr);
        operands[arg.firArgument] = [=](IterSpace iters) {
          return lambda(iters);
        };
      } break;
      case PassBy::BaseAddressValueAttribute: {
        // VALUE attribute or pass-by-reference to a copy semantics. (byval*)
        PushSemantics(ConstituentSemantics::ByValueArg);
        auto lambda = genarr(*expr);
        operands[arg.firArgument] = [=](IterSpace iters) {
          return lambda(iters);
        };
      } break;
      case PassBy::BaseAddress: {
        PushSemantics(ConstituentSemantics::RefOpaque);
        auto lambda = genarr(*expr);
        operands[arg.firArgument] = [=](IterSpace iters) {
          return lambda(iters);
        };
      } break;
      case PassBy::CharBoxValueAttribute:
        TODO(loc, "CHARACTER, VALUE");
        break;
      case PassBy::BoxChar:
        TODO(loc, "CHARACTER");
        break;
      case PassBy::AddressAndLength:
        TODO(loc, "address and length argument");
        break;
      case PassBy::Box:
      case PassBy::MutableBox:
        // See C15100 and C15101
        fir::emitFatalError(loc, "cannot be POINTER, ALLOCATABLE");
      }
    }

    if (caller.getIfIndirectCallSymbol())
      fir::emitFatalError(loc, "cannot be indirect call");
    auto funcSym = builder.getSymbolRefAttr(caller.getMangledName());
    auto resTys = caller.getFuncOp().getType().getResults();
    if (caller.getFuncOp().getType().getResults() !=
        caller.genFunctionType().getResults())
      fir::emitFatalError(loc, "type mismatch on declared function");
    return [=](IterSpace iters) -> ExtValue {
      llvm::SmallVector<mlir::Value> args;
      for (const auto &cc : operands)
        args.push_back(fir::getBase(cc(iters)));
      auto call = builder.create<fir::CallOp>(loc, resTys, funcSym, args);
      return call.getResult(0);
    };
  }

  // A procedure reference.
  CC genProcRef(const Fortran::evaluate::ProcedureRef &procRef,
                llvm::Optional<mlir::Type> retTy) {
    auto loc = getLoc();
    if (procRef.IsElemental()) {
      if (const auto *intrin = procRef.proc().GetSpecificIntrinsic()) {
        // Elemental intrinsic call.
        // The intrinsic procedure is called once per element of the array.
        return genElementalIntrinsicProcRef(procRef, retTy, *intrin);
      }
      if (ScalarExprLowering::isStatementFunctionCall(procRef))
        fir::emitFatalError(loc, "statement function cannot be elemental");

      // Elemental call.
      // The procedure is called once per element of the array argument(s).
      return genElementalUserDefinedProcRef(procRef, retTy);
    }

    // Transformational call.
    // The procedure is called once and produces a value of rank > 0.
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic()) {
      auto resultBox = ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}
                           .genIntrinsicRef(procRef, *intrinsic, retTy);
      return genarr(resultBox);
    }
    auto resultBox = ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}
                         .genProcedureRef(procRef, retTy);
    return genarr(resultBox);
  }

  CC genarr(const Fortran::evaluate::ProcedureDesignator &) {
    TODO(getLoc(), "procedure designator");
  }
  CC genarr(const Fortran::evaluate::ProcedureRef &x) {
    if (x.hasAlternateReturns())
      fir::emitFatalError(getLoc(),
                          "array procedure reference with alt-return");
    return genProcRef(x, llvm::None);
  }
  template <typename A, typename = std::enable_if_t<Fortran::common::HasMember<
                            A, Fortran::evaluate::TypelessExpression>>>
  CC genarr(const A &x) {
    auto result = asScalar(x);
    return [=](IterSpace) { return result; };
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Expr<A> &x) {
    if (isArray(x))
      return std::visit([&](const auto &e) { return genarr(e); }, x.u);
    auto result = asScalar(x);
    return [=](IterSpace) { return result; };
  }

  // Converting a value of memory bound type requires creating a temp and
  // copying the value.
  static ExtValue convertAdjustedType(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type toType,
                                      const ExtValue &exv) {
    auto lenFromBufferType = [&](mlir::Type ty) {
      return builder.create<mlir::ConstantIndexOp>(
          loc, fir::dyn_cast_ptrEleTy(ty).cast<fir::CharacterType>().getLen());
    };
    return exv.match(
        [&](const fir::CharBoxValue &cb) -> ExtValue {
          auto typeParams = fir::getTypeParams(exv);
          auto len = typeParams.size() > 0
                         ? typeParams[0]
                         : lenFromBufferType(cb.getBuffer().getType());
          auto mem =
              builder.create<fir::AllocaOp>(loc, toType, mlir::ValueRange{len});
          fir::CharBoxValue result(mem, len);
          Fortran::lower::CharacterExprHelper{builder, loc}.createAssign(
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
    assert(isArray(x));
    auto loc = getLoc();
    auto lambda = genarr(x.left());
    auto ty = converter.genType(TC1, KIND);
    return [=](IterSpace iters) -> ExtValue {
      auto exv = lambda(iters);
      auto val = fir::getBase(exv);
      if (elementTypeWasAdjusted(val.getType()))
        return convertAdjustedType(builder, loc, ty, exv);
      return builder.createConvert(loc, ty, val);
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    auto loc = getLoc();
    auto lambda = genarr(x.left());
    auto isImagPart = x.isImaginaryPart;
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lambda(iters));
      return Fortran::lower::ComplexExprHelper{builder, loc}.extractComplexPart(
          lhs, isImagPart);
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Parentheses<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = f(iters);
      auto base = fir::getBase(val);
      auto newBase =
          builder.create<fir::NoReassocOp>(loc, base.getType(), base);
      return fir::substBase(val, newBase);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = fir::getBase(f(iters));
      auto ty = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
      auto zero = builder.createIntegerConstant(loc, ty, 0);
      return builder.create<mlir::SubIOp>(loc, zero, val);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<fir::NegfOp>(loc, fir::getBase(f(iters)));
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    auto loc = getLoc();
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
    auto loc = getLoc();
    auto lambda = genarr(evEx.left());
    auto rf = genarr(evEx.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lambda(iters));
      auto right = fir::getBase(rf(iters));
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

  GENBIN(Add, Integer, mlir::AddIOp)
  GENBIN(Add, Real, mlir::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::SubIOp)
  GENBIN(Subtract, Real, mlir::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::MulIOp)
  GENBIN(Multiply, Real, mlir::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::SignedDivIOp)
  GENBIN(Divide, Real, mlir::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &x) {
    auto loc = getLoc();
    auto ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    switch (x.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return [=](IterSpace iters) -> ExtValue {
        auto lhs = fir::getBase(lf(iters));
        auto rhs = fir::getBase(rf(iters));
        return Fortran::lower::genMax(builder, loc,
                                      llvm::ArrayRef<mlir::Value>{lhs, rhs});
      };
    case Fortran::evaluate::Ordering::Less:
      return [=](IterSpace iters) -> ExtValue {
        auto lhs = fir::getBase(lf(iters));
        auto rhs = fir::getBase(rf(iters));
        return Fortran::lower::genMin(builder, loc,
                                      llvm::ArrayRef<mlir::Value>{lhs, rhs});
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
    auto loc = getLoc();
    auto ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::ComplexExprHelper{builder, loc}.createComplex(
          KIND, lhs, rhs);
    };
  }

  /// Fortran's concatenation operator `//`.
  template <int KIND>
  CC genarr(const Fortran::evaluate::Concat<KIND> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      auto *lchr = lhs.getCharBox();
      auto *rchr = rhs.getCharBox();
      if (lchr && rchr) {
        return Fortran::lower::CharacterExprHelper{builder, loc}
            .createConcatenate(*lchr, *rchr);
      }
      TODO(loc, "concat on unexpected extended values");
      return mlir::Value{};
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::SetLength<KIND> &x) {
    auto lf = genarr(x.left());
    auto rhs = fir::getBase(asScalar(x.right()));
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      return fir::CharBoxValue{lhs, rhs};
    };
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Constant<A> &x) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto arrTy = converter.genType(toEvExpr(x));
    auto globalName = Fortran::lower::LiteralNameHelper{x}.getName(builder);
    auto global = builder.getNamedGlobal(globalName);
    if (!global) {
      global = builder.createGlobalConstant(
          loc, arrTy, globalName,
          [&](Fortran::lower::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx;
            auto result = Fortran::lower::createSomeInitializerExpression(
                loc, converter, toEvExpr(x), symMap, stmtCtx);
            auto castTo =
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
      auto len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               charTy.getLen());
      return genarr(fir::CharArrayBoxValue{addr, len, extents});
    }
    return genarr(fir::ArrayBoxValue{addr, extents});
  }

  //===--------------------------------------------------------------------===//
  // A vector subscript expression may be wrapped with a cast to INTEGER*8.
  // Get rid of it here so the vector can be loaded. Add it back when
  // generating the elemental evaluation (inside the loop nest).

  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  ignoreEvConvert(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, 8>> &x) {
    return std::visit([&](const auto &v) { return ignoreEvConvert(v); }, x.u);
  }
  template <Fortran::common::TypeCategory FROM>
  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType> ignoreEvConvert(
      const Fortran::evaluate::Convert<
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>,
          FROM> &x) {
    return toEvExpr(x.left());
  }
  template <typename A>
  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  ignoreEvConvert(const A &x) {
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
    return Fortran::lower::readLowerBound(builder, getLoc(), x, dim, one);
  }

  /// Get the declared upper bound value of the array `x` in dimension `dim`.
  /// The argument `one` must be an ssa-value for the constant 1.
  mlir::Value getUBound(const ExtValue &x, unsigned dim, mlir::Value one) {
    auto loc = getLoc();
    auto lb = getLBound(x, dim, one);
    auto extent = Fortran::lower::readExtent(builder, loc, x, dim);
    auto add = builder.create<mlir::AddIOp>(loc, lb, extent);
    return builder.create<mlir::SubIOp>(loc, add, one);
  }

  /// Return the extent of the boxed array `x` in dimesion `dim`.
  mlir::Value getExtent(const ExtValue &x, unsigned dim) {
    return Fortran::lower::readExtent(builder, getLoc(), x, dim);
  }

  /// Array reference with subscripts. Since this has rank > 0, this is a form
  /// of an array section (slice).
  ///
  /// There are two "slicing" primitives that may be applied on a dimension by
  /// dimension basis: (1) triple notation and (2) vector addressing. Since
  /// dimensions can be selectively sliced, some dimensions may contain
  /// regular scalar expressions and those dimensions do not participate in
  /// the array expression evaluation.
  CC genarr(const Fortran::evaluate::ArrayRef &x) {
    llvm::SmallVector<mlir::Value> trips;
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto arrExt = [&]() {
      auto arrBase = x.base();
      if (arrBase.IsSymbol())
        return symMap.lookupSymbol(arrBase.GetFirstSymbol())
            .match(
                [=](const Fortran::lower::SymbolBox::None &) -> ExtValue {
                  fir::emitFatalError(loc, "symbol not available");
                },
                [=](const fir::AbstractBox &x) -> ExtValue {
                  fir::emitFatalError(loc, "array has incorrect symbol box");
                },
                [](const auto &x) -> ExtValue { return x; });
      llvm::SmallVector<mlir::Value> components;
      return buildComponentsPath(components, arrBase.GetComponent()).first;
    }();
    LLVM_DEBUG(llvm::dbgs() << "array: " << arrExt << '\n');
    PC pc = [=](IterSpace s) { return s; };
    for (auto sub : llvm::enumerate(x.subscript())) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::Triplet &t) {
                // Generate a slice operation for the triplet. The first and
                // second position of the triplet may be omitted, and the
                // declared lbound and/or ubound expression values,
                // respectively, should be used instead.
                if (auto optLo = t.lower())
                  trips.push_back(fir::getBase(asScalar(*optLo)));
                else
                  trips.push_back(getLBound(arrExt, sub.index(), one));
                if (auto optUp = t.upper())
                  trips.push_back(fir::getBase(asScalar(*optUp)));
                else
                  trips.push_back(getUBound(arrExt, sub.index(), one));
                trips.push_back(fir::getBase(asScalar(t.stride())));
              },
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &ie) {
                const auto &e = ie.value(); // get rid of bonus dereference
                if (isArray(e)) {
                  // vector-subscript: Use the index values as read from a
                  // vector to determine the temporary array value.
                  // Note: 9.5.3.3.3(3) specifies undefined behavior for
                  // multiple updates to any specific array element through a
                  // vector subscript with replicated values.
                  auto base = x.base();
                  ScalarExprLowering sel{loc, converter, symMap, stmtCtx};
                  auto exv = base.IsSymbol() ? sel.gen(base.GetFirstSymbol())
                                             : sel.gen(base.GetComponent());
                  auto arrExpr = ignoreEvConvert(e);
                  auto arrLoad =
                      lowerArraySubspace(converter, symMap, stmtCtx, arrExpr);
                  auto arrLd = arrLoad.getResult();
                  auto eleTy =
                      arrLd.getType().cast<fir::SequenceType>().getEleTy();
                  auto currentPC = pc;
                  auto dim = sub.index();
                  auto lb = Fortran::lower::readLowerBound(builder, loc, exv,
                                                           dim, one);
                  auto arrLdTypeParams = arrLoad.typeparams();
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    auto iter = newIters.iterVec()[dim];
                    auto resTy = adjustedArrayElementType(eleTy);
                    auto fetch = builder.create<fir::ArrayFetchOp>(
                        loc, resTy, arrLd, mlir::ValueRange{iter},
                        arrLdTypeParams);
                    auto cast = builder.createConvert(loc, idxTy, fetch);
                    auto val =
                        builder.create<mlir::SubIOp>(loc, idxTy, cast, lb);
                    newIters.setIndexValue(dim, val);
                    return newIters;
                  };
                  auto useInexactRange = [&]() {
                    // Get the range of the array in this dimension, [1:n:1].
                    trips.push_back(one);
                    trips.push_back(getExtent(arrExt, sub.index()));
                    trips.push_back(one);
                  };
                  if (const auto *sym = extractSubscriptSymbol(arrExpr)) {
                    auto symVal = symMap.lookupSymbol(*sym);
                    symVal.match(
                        [&](const fir::ArrayBoxValue &v) {
                          trips.push_back(getLBound(v, 0, one));
                          trips.push_back(getUBound(v, 0, one));
                          trips.push_back(one);
                        },
                        [&](auto) { useInexactRange(); });
                  } else {
                    useInexactRange();
                  }
                } else {
                  // A regular scalar index, which does not yield an array
                  // section. Use a degenerate slice operation `(e:undef:undef)`
                  // in this dimension as a placeholder. This does not
                  // necessarily change the rank of the original array, so the
                  // iteration space must also be extended to include this
                  // expression in this dimension to adjust to the array's
                  // declared rank.
                  auto base = x.base();
                  ScalarExprLowering sel{loc, converter, symMap, stmtCtx};
                  auto exv = base.IsSymbol() ? sel.gen(base.GetFirstSymbol())
                                             : sel.gen(base.GetComponent());
                  auto v = fir::getBase(asScalar(e));
                  trips.push_back(v);
                  auto undef = builder.create<fir::UndefOp>(loc, idxTy);
                  trips.push_back(undef);
                  trips.push_back(undef);
                  auto currentPC = pc;
                  // Cast `e` to index type.
                  auto iv = builder.createConvert(loc, idxTy, v);
                  auto dim = sub.index();
                  auto lb = Fortran::lower::readLowerBound(builder, loc, exv,
                                                           dim, one);
                  // Normalize `e` by subtracting the declared lbound.
                  mlir::Value ivAdj =
                      builder.create<mlir::SubIOp>(loc, idxTy, iv, lb);
                  // Add lbound adjusted value of `e` to the iteration vector.
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    newIters.insertIndexValue(dim, ivAdj);
                    return newIters;
                  };
                }
              }},
          sub.value().u);
    }
    auto lambda = genSlice(x.base(), trips);
    return [=](IterSpace iters) { return lambda(pc(iters)); };
  }
  CC genarr(const Fortran::evaluate::NamedEntity &entity) {
    if (entity.IsSymbol())
      return genarr(Fortran::semantics::SymbolRef{entity.GetFirstSymbol()});
    return genarr(entity.GetComponent());
  }
  CC genarr(const Fortran::semantics::SymbolRef &sym) {
    return genarr(asScalarRef(sym));
  }

  CC genarr(const ExtValue &extMemref) {
    auto loc = getLoc();
    auto memref = fir::getBase(extMemref);
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(memref.getType());
    assert(arrTy.isa<fir::SequenceType>());
    auto shape = builder.createShape(loc, extMemref);
    mlir::Value slice;
    if (inSlice) {
      slice = builder.createSlice(loc, extMemref, sliceTriple, slicePath);
      if (!slicePath.empty()) {
        auto seqTy = arrTy.cast<fir::SequenceType>();
        auto eleTy = seqTy.getEleTy();
        for (auto i = slicePath.begin(), e = slicePath.end(); i != e; ++i) {
          llvm::TypeSwitch<mlir::Type>(eleTy)
              .Case<fir::ComplexType>([&](fir::ComplexType ty) {
                eleTy = Fortran::lower::ComplexExprHelper{builder, loc}
                            .getComplexPartType(ty);
              })
              .Case<fir::RecordType>([&](fir::RecordType ty) {
                auto op = i->getDefiningOp();
                if (auto off = mlir::dyn_cast<fir::FieldIndexOp>(op)) {
                  eleTy = ty.getType(off.getFieldName());
                  return;
                }
                auto off = mlir::cast<mlir::ConstantOp>(op);
                eleTy = ty.getType(fir::toInt(off));
              })
              .Case<fir::SequenceType>([&](fir::SequenceType ty) {
                auto rank = ty.getDimension();
                if (std::distance(i, e) < rank)
                  fir::emitFatalError(loc, "slicing path is ill-formed");
                i += rank;
                eleTy = ty.getEleTy();
              })
              .Default([&](auto) {
                fir::emitFatalError(loc, "invalid slice subtype");
              });
        }
        // create the type of the projected array.
        arrTy = fir::SequenceType::get(seqTy.getShape(), eleTy);
        LLVM_DEBUG(llvm::dbgs()
                   << "type of array projection from component slicing: "
                   << eleTy << ", " << arrTy << '\n');
      }
    }
    if (isBoxValue()) {
      auto boxTy = fir::BoxType::get(reduceRank(arrTy, slice));
      if (memref.getType().isa<fir::BoxType>())
        TODO(loc, "use fir.rebox for array section of fir.box");
      mlir::Value embox = builder.create<fir::EmboxOp>(
          loc, boxTy, memref, shape, slice, /*lenParams=*/llvm::None);
      return [=](IterSpace) -> ExtValue { return fir::BoxValue(embox); };
    }
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    if (isReferentiallyOpaque()) {
      auto refEleTy = builder.getRefType(eleTy);
      return [=](IterSpace iters) -> ExtValue {
        mlir::Value coor = builder.create<fir::ArrayCoorOp>(
            loc, refEleTy, memref, shape, slice, iters.iterVec(),
            /*lenParams=*/llvm::None);
        return coor;
      };
    }
    auto arrLoad = builder.create<fir::ArrayLoadOp>(
        loc, arrTy, memref, shape, slice, fir::getTypeParams(extMemref));
    auto arrLd = arrLoad.getResult();
    if (isProjectedCopyInCopyOut()) {
      destination = arrLoad;
      return [=](IterSpace iters) -> ExtValue {
        auto innerArg = iters.innerArgument();
        auto resTy = adjustedArrayElementType(innerArg.getType());
        auto arrUpdate = builder.create<fir::ArrayUpdateOp>(
            loc, resTy, innerArg, iters.getElement(), iters.iterVec(),
            destination.typeparams());
        return arrayElementToExtendedValue(builder, loc, extMemref, arrUpdate);
      };
    }
    arrayOperandLoads.emplace_back(arrLoad);
    if (isCopyInCopyOut())
      return [=](IterSpace) -> ExtValue { return arrLd; };
    auto arrLdTypeParams = arrLoad.typeparams();
    auto resTy = adjustedArrayElementType(eleTy);
    if (isValueAttribute())
      return [=](IterSpace iters) -> ExtValue {
        auto arrFetch = builder.create<fir::ArrayFetchOp>(
            loc, resTy, arrLd, iters.iterVec(), arrLdTypeParams);
        auto base = arrFetch.getResult();
        auto temp = builder.createTemporary(
            loc, base.getType(),
            llvm::ArrayRef<mlir::NamedAttribute>{getAdaptToByRefAttr()});
        builder.create<fir::StoreOp>(loc, base, temp);
        return arrayElementToExtendedValue(builder, loc, extMemref, temp);
      };
    return [=](IterSpace iters) -> ExtValue {
      auto arrFetch = builder.create<fir::ArrayFetchOp>(
          loc, resTy, arrLd, iters.iterVec(), arrLdTypeParams);
      return arrayElementToExtendedValue(builder, loc, extMemref, arrFetch);
    };
  }

  /// Reduce the rank of a array to be boxed based on the slice's operands.
  static mlir::Type reduceRank(mlir::Type arrTy, mlir::Value slice) {
    if (slice) {
      auto slOp = mlir::dyn_cast<fir::SliceOp>(slice.getDefiningOp());
      assert(slOp);
      auto seqTy = arrTy.dyn_cast<fir::SequenceType>();
      assert(seqTy);
      auto triples = slOp.triples();
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

  /// Example: <code>array%baz%qux%waldo</code>
  CC genarr(const Fortran::evaluate::Component &x) {
    llvm::SmallVector<mlir::Value> components;
    auto pair = buildComponentsPath(components, x);
    auto lambda = genPathSlice(pair.first, components);
    return [=](IterSpace iters) { return lambda(iters); };
  }

  /// The `Ev::Component` structure is tailmost down to head, so the expression
  /// <code>a%b%c</code> will be presented as <code>(component (dataref
  /// (component (dataref (symbol 'a)) (symbol 'b))) (symbol 'c))</code>.
  std::pair<ExtValue, mlir::Type>
  buildComponentsPath(llvm::SmallVectorImpl<mlir::Value> &components,
                      const Fortran::evaluate::Component &x) {
    using RT = std::pair<ExtValue, mlir::Type>;
    auto dr = x.base();
    if (dr.Rank() == 0) {
      auto exv = asScalarRef(x);
      return RT{exv, fir::getBase(exv).getType()};
    }
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Component &c) {
              auto [exv, refTy] = buildComponentsPath(components, c);
              auto ty = fir::dyn_cast_ptrEleTy(refTy);
              assert(ty.isa<fir::SequenceType>());
              auto arrTy = ty.cast<fir::SequenceType>();
              auto name = toStringRef(x.GetLastSymbol().name());
              auto recTy = arrTy.getEleTy();
              auto eleTy = recTy.cast<fir::RecordType>().getType(name);
              auto fldTy = fir::FieldType::get(eleTy.getContext());
              components.push_back(builder.create<fir::FieldIndexOp>(
                  getLoc(), fldTy, name, recTy, fir::getTypeParams(exv)));
              return RT{exv, builder.getRefType(fir::SequenceType::get(
                                 arrTy.getShape(), eleTy))};
            },
            [&](const Fortran::semantics::SymbolRef &y) {
              auto exv = asScalarRef(y);
              return RT{exv, fir::getBase(exv).getType()};
            },
            [&](const Fortran::evaluate::ArrayRef &r) {
              // Must be scalar per C919 and C925
              auto exv = asScalarRef(r);
              return RT{exv, fir::getBase(exv).getType()};
            },
            [&](const Fortran::evaluate::CoarrayRef &r) {
              // Must be scalar per C919 and C925
              auto exv = asScalarRef(r);
              return RT{exv, fir::getBase(exv).getType()};
            }},
        dr.u);
  }

  /// Example: <code>array%RE</code>
  CC genarr(const Fortran::evaluate::ComplexPart &x) {
    auto loc = getLoc();
    auto i32Ty = builder.getI32Type(); // llvm's GEP requires i32
    auto offset = builder.createIntegerConstant(
        loc, i32Ty,
        x.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    auto lambda = genPathSlice(x.complex(), {offset});
    return [=](IterSpace iters) { return lambda(iters); };
  }

  template <typename A>
  CC genPathSlice(const A &x, mlir::ValueRange path) {
    auto saveInSlice = inSlice;
    inSlice = true;
    auto sz = slicePath.size();
    slicePath.append(path.begin(), path.end());
    auto result = genarr(x);
    slicePath.resize(sz);
    inSlice = saveInSlice;
    return result;
  }
  template <typename A>
  CC genSlice(const A &x, mlir::ValueRange trips) {
    if (sliceTriple.size() != 0)
      fir::emitFatalError(getLoc(), "multiple slices");
    auto saveInSlice = inSlice;
    inSlice = true;
    sliceTriple.append(trips.begin(), trips.end());
    auto result = genarr(x);
    sliceTriple.clear();
    inSlice = saveInSlice;
    return result;
  }

  CC genarr(const Fortran::evaluate::CoarrayRef &) {
    TODO(getLoc(), "coarray ref");
  }

  /// 9.4.1 Substrings
  CC genarr(const Fortran::evaluate::Substring &x) {
    auto loc = getLoc();
    auto pf = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &p) { return genarr(p); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &) -> CC {
              fir::emitFatalError(loc, "substring of static array object");
            }},
        x.parent());
    // lower and upper *must* be scalars
    llvm::SmallVector<mlir::Value> bounds = {fir::getBase(asScalar(x.lower()))};
    if (auto upper = x.upper())
      bounds.push_back(fir::getBase(asScalar(*upper)));
    return [=](IterSpace iters) -> ExtValue {
      auto base = pf(iters);
      if (auto *chr = base.getCharBox())
        return Fortran::lower::CharacterExprHelper{builder, loc}
            .createSubstring(*chr, bounds);
      TODO(loc, "unhandled substring base type");
      return mlir::Value{};
    };
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    return genProcRef(x, {converter.genType(TC, KIND)});
  }

  //===--------------------------------------------------------------------===//
  // Array construction
  //===--------------------------------------------------------------------===//

  // Lower the expr cases in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::Expr<A> &x, mlir::Type,
                          mlir::Value, mlir::Value, mlir::Value,
                          Fortran::lower::StatementContext &stmtCtx) {
    if (isArray(x)) {
      auto e = toEvExpr(x);
      auto sh = Fortran::evaluate::GetShape(converter.getFoldingContext(), e);
      return {lowerSomeNewArrayExpression(converter, symMap, stmtCtx, sh, e),
              /*needCopy=*/true};
    }
    return {asScalar(x), /*needCopy=*/true};
  }

  // Target agnostic computation of the size of an element in the array.
  mlir::Value computeElementSize(mlir::Type eleRefTy, mlir::Type resRefTy,
                                 mlir::Value mem) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto nullPtr = builder.createNullConstant(loc, resRefTy);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto offset = builder.create<fir::CoordinateOp>(loc, eleRefTy, nullPtr,
                                                    mlir::ValueRange{one});
    return builder.createConvert(loc, idxTy, offset);
  }

  /// Get the function signature of the LLVM memcpy intrinsic.
  mlir::FunctionType memcpyType() {
    return Fortran::lower::getLlvmMemcpy(builder).getType();
  }

  /// Create a call to the LLVM memcpy intrinsic.
  void createCallMemcpy(llvm::ArrayRef<mlir::Value> args) {
    auto loc = getLoc();
    auto memcpyFunc = Fortran::lower::getLlvmMemcpy(builder);
    auto funcSymAttr = builder.getSymbolRefAttr(memcpyFunc.getName());
    auto funcTy = memcpyFunc.getType();
    builder.create<fir::CallOp>(loc, funcTy.getResults(), funcSymAttr, args);
  }

  // Construct code to check for a buffer overrun and realloc the buffer when
  // space is depleted. This is done between each item in the ac-value-list.
  mlir::Value growBuffer(mlir::Value mem, mlir::Value needed,
                         mlir::Value bufferSize, mlir::Value buffSize,
                         mlir::Value eleSz) {
    auto loc = getLoc();
    auto reallocFunc = Fortran::lower::getRealloc(builder);
    auto cond = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sle,
                                             bufferSize, needed);
    auto ifOp = builder.create<fir::IfOp>(loc, mem.getType(), cond,
                                          /*withElseRegion=*/true);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&ifOp.thenRegion().front());
    // Not enough space, resize the buffer.
    auto idxTy = builder.getIndexType();
    auto two = builder.createIntegerConstant(loc, idxTy, 2);
    auto newSz = builder.create<mlir::MulIOp>(loc, needed, two);
    builder.create<fir::StoreOp>(loc, newSz, buffSize);
    mlir::Value byteSz = builder.create<mlir::MulIOp>(loc, newSz, eleSz);
    auto funcSymAttr = builder.getSymbolRefAttr(reallocFunc.getName());
    auto funcTy = reallocFunc.getType();
    auto newMem = builder.create<fir::CallOp>(
        loc, funcTy.getResults(), funcSymAttr,
        llvm::ArrayRef<mlir::Value>{
            builder.createConvert(loc, funcTy.getInputs()[0], mem),
            builder.createConvert(loc, funcTy.getInputs()[1], byteSz)});
    auto castNewMem =
        builder.createConvert(loc, mem.getType(), newMem.getResult(0));
    builder.create<fir::ResultOp>(loc, castNewMem);
    builder.setInsertionPointToStart(&ifOp.elseRegion().front());
    // Otherwise, just forward the buffer.
    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    return ifOp.getResult(0);
  }

  // Copy the next value (or vector of values) into the array being constructed.
  mlir::Value copyNextArrayCtorSection(const ExtValue &exv, mlir::Value buffPos,
                                       mlir::Value buffSize, mlir::Value mem,
                                       mlir::Value eleSz, mlir::Type eleTy,
                                       mlir::Type eleRefTy, mlir::Type resTy) {
    auto loc = getLoc();
    auto off = builder.create<fir::LoadOp>(loc, buffPos);
    auto limit = builder.create<fir::LoadOp>(loc, buffSize);
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);

    // Lambda to lower an abstract array box value.
    auto doAbstractArray = [&](const auto &v) {
      // Compute the array size.
      auto arrSz = one;
      for (auto ext : v.getExtents())
        arrSz = builder.create<mlir::MulIOp>(loc, arrSz, ext);

      // Grow the buffer as needed.
      auto endOff = builder.create<mlir::AddIOp>(loc, off, arrSz);
      mem = growBuffer(mem, endOff, limit, buffSize, eleSz);

      // Copy the elements to the buffer.
      mlir::Value byteSz = builder.create<mlir::MulIOp>(loc, arrSz, eleSz);
      auto buff = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
      auto buffi = builder.create<fir::CoordinateOp>(loc, eleRefTy, buff,
                                                     mlir::ValueRange{off});
      auto args = Fortran::lower::createArguments(
          builder, loc, memcpyType(), buffi, v.getAddr(), byteSz,
          /*volatile=*/builder.createBool(loc, false));
      createCallMemcpy(args);

      // Save the incremented buffer position.
      builder.create<fir::StoreOp>(loc, endOff, buffPos);
    };

    // Copy the value.
    exv.match(
        [&](const mlir::Value &v) {
          // Increment the buffer position.
          auto plusOne = builder.create<mlir::AddIOp>(loc, off, one);

          // Grow the buffer as needed.
          mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

          // Store the element in the buffer.
          auto buff =
              builder.createConvert(loc, fir::HeapType::get(resTy), mem);
          auto buffi = builder.create<fir::CoordinateOp>(loc, eleRefTy, buff,
                                                         mlir::ValueRange{off});
          auto val = builder.createConvert(loc, eleTy, v);
          builder.create<fir::StoreOp>(loc, val, buffi);

          builder.create<fir::StoreOp>(loc, plusOne, buffPos);
        },
        [&](const fir::CharBoxValue &v) {
          // Increment the buffer position.
          auto plusOne = builder.create<mlir::AddIOp>(loc, off, one);

          // Grow the buffer as needed.
          mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

          // Store the element in the buffer.
          auto buff =
              builder.createConvert(loc, fir::HeapType::get(resTy), mem);
          auto buffi = builder.create<fir::CoordinateOp>(loc, eleRefTy, buff,
                                                         mlir::ValueRange{off});
          auto castedStr = builder.createConvert(loc, eleRefTy, v.getAddr());
          auto load = builder.create<fir::LoadOp>(loc, castedStr);
          builder.create<fir::StoreOp>(loc, load, buffi);

          builder.create<fir::StoreOp>(loc, plusOne, buffPos);
        },
        [&](const fir::ArrayBoxValue &v) { doAbstractArray(v); },
        [&](const fir::CharArrayBoxValue &v) { doAbstractArray(v); },
        [&](const auto &) {
          TODO(loc, "unhandled array constructor expression");
        });
    return mem;
  }

  // Lower an ac-implied-do in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::ImpliedDo<A> &x,
                          mlir::Type resTy, mlir::Value mem,
                          mlir::Value buffPos, mlir::Value buffSize,
                          Fortran::lower::StatementContext &) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto lo =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.lower())));
    auto up =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.upper())));
    auto step =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.stride())));
    auto seqTy = resTy.template cast<fir::SequenceType>();
    auto eleTy = fir::unwrapSequenceType(seqTy);
    auto loop =
        builder.create<fir::DoLoopOp>(loc, lo, up, step, /*unordered=*/false,
                                      /*finalCount=*/false, mem);
    // create a new binding for x.name(), to ac-do-variable, to the iteration
    // value.
    symMap.pushImpliedDoBinding(toStringRef(x.name()), loop.getInductionVar());
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loop.getBody());

    auto eleRefTy = builder.getRefType(eleTy);
    auto eleSz = computeElementSize(eleRefTy, builder.getRefType(resTy), mem);

    // Cleanups for temps in loop body. Any temps created in the loop body need
    // to be freed before the end of the loop.
    Fortran::lower::StatementContext loopCtx;
    for (const Fortran::evaluate::ArrayConstructorValue<A> &acv : x.values()) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &v) {
            return genArrayCtorInitializer(v, resTy, mem, buffPos, buffSize,
                                           loopCtx);
          },
          acv.u);
      if (copyNeeded)
        mem = copyNextArrayCtorSection(exv, buffPos, buffSize, mem, eleSz,
                                       eleTy, eleRefTy, resTy);
    }
    loopCtx.finalize();

    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    mem = loop.getResult(0);
    symMap.popImpliedDoBinding();
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos).getResult()};

    // Convert to extended value.
    if (auto charTy =
            seqTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      auto len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               charTy.getLen());
      return {fir::CharArrayBoxValue{mem, len, extents}, /*needCopy=*/false};
    }
    return {fir::ArrayBoxValue{mem, extents}, /*needCopy=*/false};
  }

  // To simplify the handling and interaction between the various cases, array
  // constructors are always lowered to the incremental construction code
  // pattern, even if the extent of the array value is constant. After the
  // MemToReg pass and constant folding, the optimizer should be able to
  // determine that all the buffer overrun tests are false when the incremental
  // construction wasn't actually required.
  template <typename A>
  CC genarr(const Fortran::evaluate::ArrayConstructor<A> &x) {
    auto loc = getLoc();
    auto evExpr = toEvExpr(x);
    auto resTy = translateSomeExprToFIRType(converter, evExpr);
    auto idxTy = builder.getIndexType();
    auto seqTy = resTy.template cast<fir::SequenceType>();
    auto eleTy = fir::unwrapSequenceType(resTy);
    auto buffSize = builder.createTemporary(loc, idxTy, ".buff.size");
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto buffPos = builder.createTemporary(loc, idxTy, ".buff.pos");
    builder.create<fir::StoreOp>(loc, zero, buffPos);
    // Allocate space for the array to be constructed.
    mlir::Value mem;
    if (fir::hasDynamicSize(resTy)) {
      auto initBuffSz =
          builder.createIntegerConstant(loc, idxTy, clInitialBufferSize);
      mem = builder.create<fir::AllocMemOp>(
          loc, eleTy, /*typeparams=*/llvm::None, initBuffSz);
      builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
    } else {
      mem = builder.create<fir::AllocMemOp>(loc, resTy);
      int64_t buffSz = 1;
      for (auto extent : seqTy.getShape())
        buffSz *= extent;
      auto initBuffSz = builder.createIntegerConstant(loc, idxTy, buffSz);
      builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
    }
    // Compute size of element
    auto eleRefTy = builder.getRefType(eleTy);
    auto eleSz = computeElementSize(eleRefTy, builder.getRefType(resTy), mem);

    // Populate the buffer with the elements, growing as necessary.
    for (const auto &expr : x) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &e) {
            return genArrayCtorInitializer(e, resTy, mem, buffPos, buffSize,
                                           stmtCtx);
          },
          expr.u);
      if (copyNeeded)
        mem = copyNextArrayCtorSection(exv, buffPos, buffSize, mem, eleSz,
                                       eleTy, eleRefTy, resTy);
    }
    mem = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos)};

    // Cleanup the temporary.
    auto *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup(
        [bldr, loc, mem]() { bldr->create<fir::FreeMemOp>(loc, mem); });

    // Return the continuation.
    if (auto charTy =
            seqTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      auto len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               charTy.getLen());
      return genarr(fir::CharArrayBoxValue{mem, len, extents});
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
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lambda = genarr(x.left());
    auto truth = builder.createBool(loc, true);
    return [=](IterSpace iters) -> ExtValue {
      auto logical = fir::getBase(lambda(iters));
      auto val = builder.createConvert(loc, i1Ty, logical);
      return builder.create<mlir::XOrOp>(loc, val, truth);
    };
  }
  template <typename OP, typename A>
  CC createBinaryBoolOp(const A &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
      auto right = fir::getBase(rf(iters));
      auto lhs = builder.createConvert(loc, i1Ty, left);
      auto rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, lhs, rhs);
    };
  }
  template <typename OP, typename A>
  CC createCompareBoolOp(mlir::CmpIPredicate pred, const A &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
      auto right = fir::getBase(rf(iters));
      auto lhs = builder.createConvert(loc, i1Ty, left);
      auto rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::LogicalOperation<KIND> &x) {
    switch (x.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryBoolOp<mlir::AndOp>(x);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryBoolOp<mlir::OrOp>(x);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareBoolOp<mlir::CmpIOp>(mlir::CmpIPredicate::eq, x);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareBoolOp<mlir::CmpIOp>(mlir::CmpIPredicate::ne, x);
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
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <typename A>
  CC createCompareCharOp(mlir::CmpIPredicate pred, const A &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      return Fortran::lower::genCharCompare(builder, loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    return createCompareOp<mlir::CmpIOp>(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Character, KIND>> &x) {
    return createCompareCharOp(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    return createCompareOp<fir::CmpfOp>(translateFloatRelational(x.opr), x);
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

  //===--------------------------------------------------------------------===//
  // Boilerplate variants
  //===--------------------------------------------------------------------===//

  template <typename A>
  CC genarr(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genarr(x); }, des.u);
  }
  CC genarr(const Fortran::evaluate::DataRef &d) {
    return std::visit([&](const auto &x) { return genarr(x); }, d.u);
  }

private:
  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             fir::ArrayLoadOp dst = {})
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, destination{dst} {}

  explicit ArrayExprLowering(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::StatementContext &stmtCtx, Fortran::lower::SymMap &symMap,
      fir::ArrayLoadOp dst,
      const std::optional<Fortran::evaluate::Shape> &shape)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, destination{dst}, destShape{shape} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem,
                             fir::ArrayLoadOp dst = {})
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, destination{dst}, semant{sem} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem,
                             Fortran::lower::MaskExpr *masks)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, masks{masks}, semant{sem} {}

  mlir::Location getLoc() { return converter.getCurrentLocation(); }

  /// Array appears in a lhs context such that it is assigned after the rhs is
  /// fully evaluated.
  bool isCopyInCopyOut() {
    return semant == ConstituentSemantics::CopyInCopyOut;
  }

  /// Array appears in a lhs (or temp) context such that a projected,
  /// discontiguous subspace of the array is assigned after the rhs is fully
  /// evaluated. That is, the rhs array value is merged into a section of the
  /// lhs array.
  bool isProjectedCopyInCopyOut() {
    return semant == ConstituentSemantics::ProjectedCopyInCopyOut;
  }

  /// Array appears in a context where it must be boxed.
  bool isBoxValue() { return semant == ConstituentSemantics::BoxValue; }

  /// Array appears in a context where differences in the memory reference can
  /// be observable in the computational results. For example, an array
  /// element is passed to an impure procedure.
  bool isReferentiallyOpaque() {
    return semant == ConstituentSemantics::RefOpaque;
  }

  /// Array appears in a context where it is passed as a VALUE argument.
  bool isValueAttribute() { return semant == ConstituentSemantics::ByValueArg; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  llvm::Optional<CC> ccDest;
  fir::ArrayLoadOp destination;
  std::optional<Fortran::evaluate::Shape> destShape;
  llvm::SmallVector<fir::ArrayLoadOp> arrayOperandLoads;
  llvm::SmallVector<mlir::Value> sliceTriple;
  llvm::SmallVector<mlir::Value> slicePath;
  Fortran::lower::MaskExpr *masks{};
  ConstituentSemantics semant{ConstituentSemantics::RefTransparent};
  bool inSlice{false};
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.genExtValue(expr);
}

fir::ExtendedValue Fortran::lower::createSomeInitializerExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx,
                            /*initializer=*/true}
      .genExtValue(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(expr);
}

void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ") << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}

void Fortran::lower::createMaskedArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::MaskExpr &masks, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given mask conditions\n";);
  ArrayExprLowering::lowerMaskedArrayAssignment(converter, symMap, stmtCtx, lhs,
                                                rhs, masks);
}

fir::ExtendedValue Fortran::lower::createSomeArrayTempValue(
    Fortran::lower::AbstractConverter &converter,
    const std::optional<Fortran::evaluate::Shape> &shape,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  return ArrayExprLowering::lowerSomeNewArrayExpression(converter, symMap,
                                                        stmtCtx, shape, expr);
}

fir::ExtendedValue Fortran::lower::createSomeArrayBox(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "box designator: ") << '\n');
  return ArrayExprLowering::lowerArrayExpressionBoxed(converter, symMap,
                                                      stmtCtx, expr);
}

fir::MutableBoxValue Fortran::lower::createSomeMutableBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  // MutableBox lowering StatementContext does not need to be propagated
  // to the caller because the result value is a variable, not a temporary
  // expression. The StatementContext clean-up can occur before using the
  // resulting MutableBoxValue. Variables of all other types are handled in the
  // bridge.
  Fortran::lower::StatementContext dummyStmtCtx;
  return ScalarExprLowering{loc, converter, symMap, dummyStmtCtx}
      .genMutableBoxValue(expr);
}
