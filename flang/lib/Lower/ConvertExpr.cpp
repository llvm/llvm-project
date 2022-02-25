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
#include "flang/Lower/DumpEvaluateExpr.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
                              Fortran::lower::StatementContext &stmtCtx)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap} {
  }

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
      return val.match([&val](auto &) { return val.toExtendedValue(); });
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
    TODO(getLoc(), "genval ImpliedDoIndex");
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    TODO(getLoc(), "genval DescriptorInquiry");
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
    TODO(getLoc(), "genval Power");
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    TODO(getLoc(), "genval RealToInt");
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

  /// Convert a ascii scalar literal CHARACTER to IR. (specialization)
  ExtValue
  genAsciiScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                        Fortran::common::TypeCategory::Character, 1>> &value,
                    int64_t len) {
    assert(value.size() == static_cast<std::uint64_t>(len) &&
           "value.size() doesn't match with len");
    return fir::factory::createStringLiteral(builder, getLoc(), value);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    if (con.Rank() > 0)
      TODO(getLoc(), "genval array constant");
    std::optional<Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>>
        opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      if constexpr (KIND == 1)
        return genAsciiScalarLit(opt.value(), con.LEN());
      TODO(getLoc(), "genval for Character with KIND != 1");
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
        TODO(getLoc(), "genOffsetAndCoordinateOp");
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

  ExtValue gen(const Fortran::evaluate::ArrayRef &aref) {
    ExtValue base = aref.base().IsSymbol() ? gen(aref.base().GetFirstSymbol())
                                           : gen(aref.base().GetComponent());
    return genCoordinateOp(base, aref);
  }
  ExtValue genval(const Fortran::evaluate::ArrayRef &aref) {
    TODO(getLoc(), "genval ArrayRef");
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
    TODO(getLoc(), "gen FunctionRef<A>");
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
    if (caller.getIfIndirectCallSymbol()) {
      TODO(loc, "genCallOpAndResult indirect call");
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
        TODO(loc, "genCallOpResult extents and length");
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
              TODO(loc, "allocatedResult for allocatable");
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
        TODO(loc, "arg passby MutableBox");
      }
      const bool actualArgIsVariable = Fortran::evaluate::IsVariable(*expr);
      if (arg.passBy == PassBy::BaseAddress || arg.passBy == PassBy::BoxChar) {
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
            TODO(loc, "procedureref arrays");
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
          TODO(loc, "procedureref PassBy::BoxChar");
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
      switch (argRules.lowerAs) {
      case Fortran::lower::LowerIntrinsicArgAs::Value:
        operands.emplace_back(genval(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Addr:
        TODO(getLoc(), "argument lowering for Addr");
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Box:
        TODO(getLoc(), "argument lowering for Box");
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Inquired:
        TODO(getLoc(), "argument lowering for Inquired");
        continue;
      }
      llvm_unreachable("bad switch");
    }
    // Let the intrinsic library lower the intrinsic procedure call
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name, resultType,
                                            operands);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x))
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    TODO(getLoc(), "genval Expr<A> arrays");
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
  ExtValue gen(const Fortran::evaluate::Expr<A> &x) {
    // Whole array symbols or components, and results of transformational
    // functions already have a storage and the scalar expression lowering path
    // is used to not create a new temporary storage.
    if (isScalar(x) ||
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(x) ||
        isTransformationalRef(x))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
    TODO(getLoc(), "gen Expr non-scalar");
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

  // An expression with non-zero rank is an array expression.
  template <typename A>
  bool isArray(const A &x) const {
    return x.Rank() != 0;
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

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  CC genarr(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type<       \
                Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) {       \
    TODO(getLoc(), "genarr Binary");                                           \
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
    TODO(getLoc(), "genarr ");
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    TODO(getLoc(), "genarr ");
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    TODO(getLoc(), "genarr ");
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    TODO(getLoc(), "genarr ");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::Concat<KIND> &x) {
    TODO(getLoc(), "genarr ");
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::SetLength<KIND> &x) {
    TODO(getLoc(), "genarr ");
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Constant<A> &x) {
    TODO(getLoc(), "genarr ");
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

  template <typename A>
  CC genarr(const Fortran::evaluate::ArrayConstructor<A> &x) {
    TODO(getLoc(), "genarr ArrayConstructor<A>");
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
    TODO(getLoc(), "genarr FunctionRef");
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

  CC genarr(const Fortran::evaluate::ArrayRef &x, ComponentPath &components) {
    TODO(getLoc(), "genar  ArrayRef");
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
      TODO(loc, "genarr with Slices");
    }
    arrayOperands.push_back(ArrayOperand{memref, shape, slice});
    if (destShape.empty())
      destShape = getShape(arrayOperands.back());
    if (isBoxValue()) {
      TODO(loc, "genarr BoxValue");
    }
    if (isReferentiallyOpaque()) {
      TODO(loc, "genarr isReferentiallyOpaque");
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
      TODO(loc, "isCustomCopyInCopyOut");
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
            // mlir::Value dstLen = fir::factory::genLenOfCharacter(
            //     builder, loc, arrLoad, iters.iterVec(), substringBounds);
            // fir::CharBoxValue dstChar(arrayOp, dstLen);
            // return fir::factory::CharacterExprHelper{builder, loc}
            //     .createSubstring(dstChar, substringBounds);
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
        TODO(loc, "isAdjustedArrayElementType");
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

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(expr);
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
