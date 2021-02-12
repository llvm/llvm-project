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

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ScalarExprLowering {
public:
  explicit ScalarExprLowering(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              Fortran::lower::SymMap &symMap,
                              Fortran::lower::StatementContext &stmtCtx,
                              bool initializer = false)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap},
        inInitializer{initializer} {}

  fir::ExtendedValue genExtAddr(const Fortran::lower::SomeExpr &expr) {
    return gen(expr);
  }

  /// Lower `expr` to be passed as an argument. The expression is passed by
  /// reference.
  fir::ExtendedValue genExtArg(const Fortran::lower::SomeExpr &expr,
                               bool mayUseBox) {
    bool saveUseBoxArg = useBoxArg;
    useBoxArg = mayUseBox;
    auto result = genExtAddr(expr);
    useBoxArg = saveUseBoxArg;
    return result;
  }

  fir::ExtendedValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  fir::ExtendedValue genStringLit(llvm::StringRef str, std::uint64_t len) {
    return genScalarLit<1>(str.str(), static_cast<int64_t>(len));
  }

  fir::MutableBoxValue
  genMutableBoxValue(const Fortran::lower::SomeExpr &expr) {
    // TODO: GetLastSymbol is not the right thing to do if expr if an
    // allocatable or pointer derived type component.
    auto *sym = Fortran::evaluate::GetLastSymbol(expr);
    if (!sym)
      fir::emitFatalError(getLoc(), "trying to get descriptor address of an "
                                    "expression that is not a variable");
    return symMap.lookupSymbol(*sym).match(
        [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &boxAddr)
            -> fir::MutableBoxValue { return boxAddr; },
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
  mlir::Value createCompareOp(mlir::CmpIPredicate pred,
                              const fir::ExtendedValue &left,
                              const fir::ExtendedValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    TODO("");
  }
  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::CmpIPredicate pred) {
    return createCompareOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  template <typename OpTy>
  mlir::Value createFltCmpOp(mlir::CmpFPredicate pred,
                             const fir::ExtendedValue &left,
                             const fir::ExtendedValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    TODO("");
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  /// Create a call to the runtime to compare two CHARACTER values.
  /// Precondition: This assumes that the two values have `fir.boxchar` type.
  mlir::Value createCharCompare(mlir::CmpIPredicate pred,
                                const fir::ExtendedValue &left,
                                const fir::ExtendedValue &right) {
    return Fortran::lower::genCharCompare(converter, getLoc(), pred, left,
                                          right);
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
  fir::ExtendedValue gen(Fortran::semantics::SymbolRef sym) {
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
  fir::ExtendedValue genLoad(const fir::ExtendedValue &addr) {
    auto loc = getLoc();
    return addr.match(
        [](const fir::CharBoxValue &box) -> fir::ExtendedValue { return box; },
        [&](const fir::UnboxedValue &v) -> fir::ExtendedValue {
          return builder.create<fir::LoadOp>(loc, fir::getBase(v));
        },
        [&](const auto &v) -> fir::ExtendedValue {
          TODO("loading array or descriptor");
        });
  }

  fir::ExtendedValue genval(Fortran::semantics::SymbolRef sym) {
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

  fir::ExtendedValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO("BOZ");
  }
  /// Return indirection to function designated in ProcedureDesignator.
  /// The type of the function indirection is not guaranteed to match the one
  /// of the ProcedureDesignator due to Fortran implicit typing rules.
  fir::ExtendedValue
  genval(const Fortran::evaluate::ProcedureDesignator &proc) {
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
  fir::ExtendedValue genval(const Fortran::evaluate::NullPointer &) {
    return builder.createNullConstant(getLoc());
  }
  fir::ExtendedValue genval(const Fortran::evaluate::StructureConstructor &) {
    TODO("struct ctor");
  }
  fir::ExtendedValue genval(const Fortran::evaluate::ImpliedDoIndex &) {
    TODO("implied do index");
  }

  fir::ExtendedValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    auto symBox = symMap.lookupSymbol(desc.base().GetLastSymbol());
    assert(symBox && "no SymbolBox associated to Symbol");
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      return symBox.getCharLen().getValue();
    default:
      TODO("descriptor inquiry other than length");
    }
    llvm_unreachable("unknown descriptor inquiry");
  }

  fir::ExtendedValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO("");
  }

  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return Fortran::lower::ComplexExprHelper{builder, getLoc()}
        .extractComplexPart(cplx, isImagPart);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    return extractComplexPart(genunbox(part.left()), part.isImaginaryPart);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
             Fortran::common::TypeCategory::Integer, KIND>> &op) {
    auto input = genunbox(op.left());
    // Like LLVM, integer negation is the binary op "0 - value"
    auto zero = genIntegerConstant<KIND>(builder.getContext(), 0);
    return builder.create<mlir::SubIOp>(getLoc(), zero, input);
  }
  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
             Fortran::common::TypeCategory::Real, KIND>> &op) {
    return builder.create<fir::NegfOp>(getLoc(), genunbox(op.left()));
  }
  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
             Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return builder.create<fir::NegcOp>(getLoc(), genunbox(op.left()));
  }

  template <typename OpTy>
  mlir::Value createBinaryOp(const fir::ExtendedValue &left,
                             const fir::ExtendedValue &right) {
    auto *lhs = left.getUnboxed();
    auto *rhs = right.getUnboxed();
    assert(lhs && rhs);
    return builder.create<OpTy>(getLoc(), *lhs, *rhs);
  }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    return createBinaryOp<OpTy>(genval(ex.left()), genval(ex.right()));
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  fir::ExtendedValue genval(                                                   \
      const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type<             \
          Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) {             \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::AddIOp)
  GENBIN(Add, Real, fir::AddfOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::SubIOp)
  GENBIN(Subtract, Real, fir::SubfOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::MulIOp)
  GENBIN(Multiply, Real, fir::MulfOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::SignedDivIOp)
  GENBIN(Divide, Real, fir::DivfOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    return Fortran::lower::ComplexExprHelper{builder, getLoc()}.createComplex(
        KIND, genunbox(op.left()), genunbox(op.right()));
  }

  template <int KIND>
  fir::ExtendedValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    auto lhs = genval(op.left());
    auto rhs = genval(op.right());
    auto *lhsChar = lhs.getCharBox();
    auto *rhsChar = rhs.getCharBox();
    if (lhsChar && rhsChar)
      return Fortran::lower::CharacterExprHelper{builder, getLoc()}
          .createConcatenate(*lhsChar, *rhsChar);
    fir::emitFatalError(getLoc(), "TODO: character array concatenate");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
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
  fir::ExtendedValue genval(const Fortran::evaluate::SetLength<KIND> &) {
    TODO("");
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
             Fortran::common::TypeCategory::Integer, KIND>> &op) {
    return createCompareOp<mlir::CmpIOp>(op, translateRelational(op.opr));
  }
  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
             Fortran::common::TypeCategory::Real, KIND>> &op) {
    return createFltCmpOp<fir::CmpfOp>(op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
             Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return createFltCmpOp<fir::CmpcOp>(op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
             Fortran::common::TypeCategory::Character, KIND>> &op) {
    return createCharCompare(op, translateRelational(op.opr));
  }

  fir::ExtendedValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    auto ty = converter.genType(TC1, KIND);
    auto operand = genunbox(convert.left());
    return builder.convertWithSemantics(getLoc(), ty, operand);
  }

  template <typename A>
  fir::ExtendedValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    auto input = genval(op.left());
    auto base = fir::getBase(input);
    mlir::Value newBase =
        builder.create<fir::NoReassocOp>(getLoc(), base.getType(), base);
    return fir::substBase(input, newBase);
  }

  template <int KIND>
  fir::ExtendedValue genval(const Fortran::evaluate::Not<KIND> &op) {
    auto logical = genunbox(op.left());
    auto one = genBoolConstant(true);
    auto val = builder.createConvert(getLoc(), builder.getI1Type(), logical);
    return builder.create<mlir::XOrOp>(getLoc(), val, one);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
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
  fir::ExtendedValue genScalarLit(
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
  /// Convert a scalar literal CHARACTER to IR. (specialization)
  template <int KIND>
  fir::ExtendedValue
  genScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                   Fortran::common::TypeCategory::Character, KIND>> &value,
               int64_t len) {
    auto type = fir::CharacterType::get(builder.getContext(), KIND, len);
    auto consLit = [&]() -> fir::StringLitOp {
      auto context = builder.getContext();
      mlir::Attribute strAttr;
      if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                   std::string>) {
        strAttr = mlir::StringAttr::get(context, value);
      } else {
        using ET = typename std::decay_t<decltype(value)>::value_type;
        std::int64_t size = static_cast<std::int64_t>(value.size());
        auto shape = mlir::VectorType::get(
            llvm::ArrayRef<std::int64_t>{size},
            mlir::IntegerType::get(builder.getContext(), sizeof(ET) * 8));
        strAttr = mlir::DenseElementsAttr::get(
            shape, llvm::ArrayRef<ET>{value.data(), value.size()});
      }
      auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), context);
      mlir::NamedAttribute dataAttr(valTag, strAttr);
      auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), context);
      mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
      llvm::SmallVector<mlir::NamedAttribute, 2> attrs{dataAttr, sizeAttr};
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
        converter.uniqueCGIdent("cl", (const char *)value.c_str());
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
  fir::ExtendedValue genArrayLit(
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &con) {
    llvm::SmallVector<mlir::Value, 8> lbounds;
    llvm::SmallVector<mlir::Value, 8> extents;
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
        llvm::SmallVector<mlir::Value, 8> idx;
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
      llvm::SmallVector<mlir::Value, 8> rangeStartIdx;
      Fortran::evaluate::ConstantSubscripts rangeStartSubscripts;
      uint64_t elemsInRange = 0;
      const uint64_t minRangeSize = 2;

      do {
        auto constant =
            fir::getBase(genScalarLit<TC, KIND>(con.At(subscripts)));
        auto createIndexes = [&](Fortran::evaluate::ConstantSubscripts subs) {
          llvm::SmallVector<mlir::Value, 8> idx;
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
            llvm::SmallVector<mlir::Value, 8> zippedRange;
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
  fir::ExtendedValue
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
  fir::ExtendedValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeKind<TC>> &con) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      auto opt = (*con).ToInt64();
      auto type = getSomeKindInteger();
      auto attr = builder.getIntegerAttr(type, opt);
      auto res = builder.create<mlir::ConstantOp>(getLoc(), type, attr);
      return res.getResult();
    } else {
      fir::emitFatalError(getLoc(), "unhandled constant of unknown kind");
    }
  }

  template <typename A>
  fir::ExtendedValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    fir::emitFatalError(getLoc(), "array constructor has no rank");
  }

  fir::ExtendedValue gen(const Fortran::evaluate::ComplexPart &x) {
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
  fir::ExtendedValue genval(const Fortran::evaluate::ComplexPart &x) {
    return genLoad(gen(x));
  }

  /// Reference to a substring.
  fir::ExtendedValue gen(const Fortran::evaluate::Substring &s) {
    // Get base string
    auto baseString = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &x) { return gen(x); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &)
                -> fir::ExtendedValue { TODO(""); },
        },
        s.parent());
    llvm::SmallVector<mlir::Value, 2> bounds;
    auto lower = genunbox(s.lower());
    bounds.push_back(lower);
    if (auto upperBound = s.upper()) {
      auto upper = genunbox(*upperBound);
      bounds.push_back(upper);
    }
    Fortran::lower::CharacterExprHelper charHelper{builder, getLoc()};
    return baseString.match(
        [&](const fir::CharBoxValue &x) -> fir::ExtendedValue {
          return charHelper.createSubstring(x, bounds);
        },
        [&](const fir::CharArrayBoxValue &) -> fir::ExtendedValue {
          // TODO: substring array
          TODO("array substring lowering");
        },
        [&](const auto &) -> fir::ExtendedValue {
          fir::emitFatalError(getLoc(), "substring base is not a CharBox");
        });
  }

  /// The value of a substring.
  fir::ExtendedValue genval(const Fortran::evaluate::Substring &ss) {
    // FIXME: why is the value of a substring being lowered the same as the
    // address of a substring?
    return gen(ss);
  }

  fir::RangeBoxValue genTriple(const Fortran::evaluate::Triplet &trip) {
    mlir::Value lower;
    if (auto lo = trip.lower())
      lower = genunbox(*lo);
    mlir::Value upper;
    if (auto up = trip.upper())
      upper = genunbox(*up);
    return {lower, upper, genunbox(trip.stride())};
  }

  /// Special factoring to allow RangeBoxValue to be returned when generating
  /// values.
  std::variant<fir::ExtendedValue, fir::RangeBoxValue>
  genComponent(const Fortran::evaluate::Subscript &subs) {
    if (auto *s = std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
            &subs.u))
      return {genval(s->value())};
    if (auto *s = std::get_if<Fortran::evaluate::Triplet>(&subs.u))
      return {genTriple(*s)};
    llvm_unreachable("unknown subscript case");
  }

  fir::ExtendedValue genval(const Fortran::evaluate::Subscript &subs) {
    if (auto *s = std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
            &subs.u))
      return {genval(s->value())};
    llvm_unreachable("unhandled subscript case");
  }

  fir::ExtendedValue gen(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  fir::ExtendedValue genval(const Fortran::evaluate::DataRef &dref) {
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
  fir::ExtendedValue gen(const Fortran::evaluate::Component &cmpt) {
    std::list<const Fortran::evaluate::Component *> list;
    auto *base = reverseComponents(cmpt, list);
    llvm::SmallVector<mlir::Value, 4> coorArgs;
    auto obj = gen(*base);
    const auto *sym = &cmpt.GetFirstSymbol();
    auto ty = converter.genType(*sym);
    auto loc = getLoc();
    auto fldTy = fir::FieldType::get(&converter.getMLIRContext());
    // FIXME: need to thread the LEN type parameters here.
    for (auto *field : list) {
      auto recTy = ty.cast<fir::RecordType>();
      sym = &field->GetLastSymbol();
      auto name = toStringRef(sym->name());
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(
          loc, fldTy, name, mlir::TypeAttr::get(recTy),
          /*lenparams=*/mlir::ValueRange{}));
      ty = recTy.getType(name);
    }
    assert(sym && "no component(s)?");
    ty = builder.getRefType(ty);
    return fir::substBase(obj, builder.create<fir::CoordinateOp>(
                                   loc, ty, fir::getBase(obj), coorArgs,
                                   /*lenParams=*/mlir::ValueRange{}));
  }

  fir::ExtendedValue genval(const Fortran::evaluate::Component &cmpt) {
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
  fir::ExtendedValue genval(const Fortran::semantics::Bound &bound) {
    if (bound.isExplicit()) {
      auto sub = bound.GetExplicit();
      if (sub.has_value())
        return genval(*sub);
      return genIntegerConstant<8>(builder.getContext(), 1);
    }
    TODO("");
  }

  fir::ExtendedValue
  genArrayRefComponent(const Fortran::evaluate::ArrayRef &aref) {
    auto base = fir::getBase(gen(aref.base().GetComponent()));
    llvm::SmallVector<mlir::Value, 8> args;
    for (auto &subsc : aref.subscript())
      args.push_back(genunbox(subsc));
    auto ty = genSubType(base.getType(), args.size());
    ty = builder.getRefType(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
  }

  static bool isSlice(const Fortran::evaluate::ArrayRef &aref) {
    for (auto &sub : aref.subscript())
      if (std::holds_alternative<Fortran::evaluate::Triplet>(sub.u))
        return true;
    return false;
  }

  fir::ExtendedValue gen(const Fortran::lower::SymbolBox &si,
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
        if (auto *trip = std::get_if<fir::RangeBoxValue>(&subVal)) {
          TODO("");
        } else {
          auto *v = std::get_if<fir::ExtendedValue>(&subVal);
          assert(v);
          if (auto *sval = v->getUnboxed()) {
            auto val = builder.createConvert(loc, idxTy, *sval);
            auto lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
            auto diff = builder.create<mlir::SubIOp>(loc, val, lb);
            auto prod = builder.create<mlir::MulIOp>(loc, delta, diff);
            total = builder.create<mlir::AddIOp>(loc, prod, total);
            if (ext)
              delta = builder.create<mlir::MulIOp>(loc, delta, ext);
          } else {
            TODO("");
          }
        }
        ++dim;
      }
      return builder.create<fir::CoordinateOp>(
          loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
    };
    return si.match(
        [&](const Fortran::lower::SymbolBox::FullDim &arr)
            -> fir::ExtendedValue {
          // FIXME: this check can be removed when slicing is implemented
          if (isSlice(aref))
            fir::emitFatalError(
                getLoc(),
                "slice should be handled in array expression context");
          return genFullDim(arr, one);
        },
        [&](const Fortran::lower::SymbolBox::CharFullDim &arr)
            -> fir::ExtendedValue {
          auto delta = arr.getLen();
          // If the length is known in the type, fir.coordinate_of will
          // already take the length into account.
          if (Fortran::lower::CharacterExprHelper::hasConstantLengthInType(arr))
            delta = one;
          return fir::CharBoxValue(genFullDim(arr, delta), arr.getLen());
        },
        [&](const Fortran::lower::SymbolBox::Derived &arr)
            -> fir::ExtendedValue {
          // TODO: implement
          mlir::emitError(loc, "not implemented: array of derived type");
          return {};
        },
        [&](const auto &) -> fir::ExtendedValue {
          mlir::emitError(loc, "internal: array lowering failed");
          return {};
        });
  }

  fir::ExtendedValue genArrayCoorOp(const fir::ExtendedValue &exv,
                                    const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = fir::getBase(exv);
    auto arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto refTy = builder.getRefType(eleTy);
    auto idxTy = builder.getIndexType();
    auto genWithShape = [&](const auto &arr) -> mlir::Value {
      auto shape = builder.consShape(loc, arr);
      llvm::SmallVector<mlir::Value, 8> arrayCoorArgs;
      for (const auto &sub : aref.subscript()) {
        auto subVal = genComponent(sub);
        if (auto *ev = std::get_if<fir::ExtendedValue>(&subVal)) {
          if (auto *sval = ev->getUnboxed()) {
            auto val = builder.createConvert(loc, idxTy, *sval);
            arrayCoorArgs.push_back(val);
          } else {
            TODO("");
          }
        } else {
          // RangedBoxValue
          TODO("");
        }
      }
      return builder.create<fir::ArrayCoorOp>(
          loc, refTy, addr, shape, mlir::Value{}, arrayCoorArgs, ValueRange());
    };
    return exv.match(
        [&](const fir::ArrayBoxValue &arr) {
          // FIXME: this check can be removed when slicing is implemented
          if (isSlice(aref))
            llvm::report_fatal_error(
                "slicing should be handled in array expresion context");
          return genWithShape(arr);
        },
        [&](const fir::CharArrayBoxValue &arr) {
          TODO("");
          return mlir::Value{};
        },
        [&](const fir::BoxValue &arr) {
          TODO("");
          return mlir::Value{};
        },
        [&](const auto &) {
          TODO("");
          return mlir::Value{};
        });
  }

  // Return the coordinate of the array reference
  fir::ExtendedValue gen(const Fortran::evaluate::ArrayRef &aref) {
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
      if (!si.hasConstantShape())
        return gen(si, aref);
      auto box = gen(symbol);
      auto base = fir::getBase(box);
      assert(base && "boxed type not handled");
      unsigned i = 0;
      llvm::SmallVector<mlir::Value, 8> args;
      auto loc = getLoc();
      for (auto &subsc : aref.subscript()) {
        auto subBox = genComponent(subsc);
        if (auto *v = std::get_if<fir::ExtendedValue>(&subBox)) {
          if (auto *val = v->getUnboxed()) {
            auto ty = val->getType();
            auto adj = getLBound(si, i++, ty);
            assert(adj && "boxed value not handled");
            args.push_back(builder.create<mlir::SubIOp>(loc, ty, *val, adj));
          } else {
            TODO("");
          }
        } else {
          auto *range = std::get_if<fir::RangeBoxValue>(&subBox);
          assert(range && "must be a range");
          // triple notation for slicing operation
          TODO("");
        }
      }
      auto ty = genSubType(base.getType(), args.size());
      ty = builder.getRefType(ty);
      auto addr = builder.create<fir::CoordinateOp>(loc, ty, base, args);
      // FIXME: return may not be a scalar.
      return box.match(
          [&](const fir::CharArrayBoxValue &x) -> fir::ExtendedValue {
            return fir::CharBoxValue{addr, x.getLen()};
          },
          [&](const auto &) -> fir::ExtendedValue { return addr; });
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

  fir::ExtendedValue genval(const Fortran::evaluate::ArrayRef &aref) {
    return genLoad(gen(aref));
  }

  fir::ExtendedValue gen(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genAddr(coref);
  }

  fir::ExtendedValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genValue(coref);
  }

  template <typename A>
  fir::ExtendedValue gen(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  fir::ExtendedValue genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  mlir::Type genType(const Fortran::evaluate::DynamicType &dt) {
    if (dt.category() != Fortran::common::TypeCategory::Derived)
      return converter.genType(dt.category(), dt.kind());
    llvm::report_fatal_error("derived types not implemented");
  }

  /// Apply the function `func` and return a reference to the resultant value.
  /// This is required for lowering expressions such as `f1(f2(v))`.
  template <typename A>
  fir::ExtendedValue gen(const Fortran::evaluate::FunctionRef<A> &func) {
    if (!func.GetType().has_value())
      mlir::emitError(getLoc(), "internal: a function must have a type");
    auto resTy = genType(*func.GetType());
    auto retVal = genProcedureRef(func, llvm::ArrayRef<mlir::Type>{resTy});
    auto retValBase = fir::getBase(retVal);
    if (fir::isa_ref_type(retValBase.getType()))
      return retVal;
    auto mem = builder.create<fir::AllocaOp>(getLoc(), retValBase.getType());
    builder.create<fir::StoreOp>(getLoc(), retValBase, mem);
    return fir::substBase(retVal, mem.getResult());
  }

  /// Generate a call to an intrinsic function.
  fir::ExtendedValue
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  mlir::ArrayRef<mlir::Type> resultTypes) {
    llvm::Optional<mlir::Type> resultType;
    if (resultTypes.size() == 1)
      resultType = resultTypes[0];

    llvm::SmallVector<fir::ExtendedValue, 2> operands;
    // Lower arguments
    // For now, logical arguments for intrinsic are lowered to `fir.logical`
    // so that TRANSFER can work. For some arguments, it could lead to useless
    // conversions (e.g scalar MASK of MERGE will be converted to `i1`), but
    // the generated code is at least correct. To improve this, the intrinsic
    // lowering facility should control argument lowering.
    for (const auto &arg : procRef.arguments()) {
      if (auto *expr = Fortran::evaluate::UnwrapExpr<
              Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg))
        operands.emplace_back(genval(*expr));
      else
        operands.emplace_back(fir::UnboxedValue{}); // absent optional
    }
    // Let the intrinsic library lower the intrinsic procedure call
    llvm::StringRef name = intrinsic.name;
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name, resultType,
                                            operands);
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
  fir::ExtendedValue
  genStmtFunctionRef(const Fortran::evaluate::ProcedureRef &procRef,
                     mlir::ArrayRef<mlir::Type> resultType) {
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
      symMap.addSymbol(*arg, genExtAddr(*expr));
    }
    auto result = genval(details.stmtFunction().value());
    LLVM_DEBUG(llvm::dbgs() << "stmt-function: " << result << '\n');
    symMap.popScope();
    return result;
  }

  fir::ExtendedValue
  genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                  mlir::ArrayRef<mlir::Type> resultType) {
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType);

    if (isStatementFunctionCall(procRef))
      return genStmtFunctionRef(procRef, resultType);

    auto loc = getLoc();
    Fortran::lower::CallerInterface caller(procRef, converter);
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    llvm::SmallVector<fir::MutableBoxValue, 1> mutableModifiedByCall;

    for (const auto &arg : caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      if (!actual)
        TODO("optional argument lowering");
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO("assumed type actual argument lowering");

      if (arg.passBy == PassBy::Value) {
        auto *argVal = genExtValue(*expr).getUnboxed();
        if (!argVal)
          mlir::emitError(
              getLoc(),
              "Lowering internal error: passing non trivial value by value");
        else
          caller.placeInput(arg, *argVal);
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
        caller.placeInput(arg, fir::getBase(argRef));
      } else if (arg.passBy == PassBy::BoxChar) {
        auto boxChar = argRef.match(
            [&](const fir::CharBoxValue &x) { return helper.createEmbox(x); },
            [&](const fir::CharArrayBoxValue &x) {
              return helper.createEmbox(x);
            },
            [&](const fir::BoxValue &x) -> mlir::Value {
              // Beware, descriptor content might have to be copied before
              // and after the call to a contiguous character argument.
              TODO("lowering actual arguments descriptor to boxchar");
            },
            [&](const auto &x) {
              mlir::emitError(loc, "Lowering internal error: actual "
                                   "argument is not a character");
              return mlir::Value{};
            });
        caller.placeInput(arg, boxChar);
      } else if (arg.passBy == PassBy::Box) {
        caller.placeInput(arg, builder.createBox(getLoc(), argRef));
      } else if (arg.passBy == PassBy::AddressAndLength) {
        caller.placeAddressAndLengthInput(arg, fir::getBase(argRef),
                                          fir::getLen(argRef));
      } else {
        llvm_unreachable("pass by value not handled here");
      }
    }

    // Handle case where caller must pass result
    auto resRef = [&]() -> llvm::Optional<fir::ExtendedValue> {
      if (auto resultArg = caller.getPassedResult()) {
        if (resultArg->passBy == PassBy::AddressAndLength) {
          // allocate and pass character result
          auto len = caller.getResultLength();
          Fortran::lower::CharacterExprHelper helper{builder, loc};
          auto temp = helper.createCharacterTemp(resultType[0], len);
          caller.placeAddressAndLengthInput(*resultArg, temp.getBuffer(),
                                            temp.getLen());
          return fir::ExtendedValue(temp);
        }
        TODO("passing hidden descriptor for result"); // Pass descriptor
      }
      return {};
    }();

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
      auto callSiteType = caller.genFunctionType();
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
        // Implicit interface result type mismatch are not standard Fortran,
        // but some compilers are not complaining about it.
        // The front-end is not protecting lowering from this currently. Support
        // this with a discouraging warning.
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
    auto funcType =
        funcPointer ? caller.genFunctionType() : caller.getFuncOp().getType();
    llvm::SmallVector<mlir::Value, 8> operands;
    // First operand of indirect call is the function pointer. Cast it to
    // required function type for the call to handle procedures that have a
    // compatible interface in Fortran, but that have different signatures in
    // FIR.
    if (funcPointer)
      operands.push_back(builder.createConvert(loc, funcType, funcPointer));

    // Deal with potential mismatches in arguments types. Passing an array to
    // a scalar argument should for instance be tolerated here.
    for (auto [fst, snd] :
         llvm::zip(caller.getInputs(), funcType.getInputs())) {
      auto cast = builder.convertWithSemantics(getLoc(), snd, fst);
      operands.push_back(cast);
    }

    auto call = builder.create<fir::CallOp>(loc, funcType.getResults(),
                                            funcSymbolAttr, operands);
    // Sync pointers and allocatables that may have been modified the call.
    for (const auto &mutableBox : mutableModifiedByCall)
      Fortran::lower::syncMutableBoxFromIRBox(builder, loc, mutableBox);
    // Handle case where result was passed as argument
    if (caller.getPassedResult())
      return resRef.getValue();

    if (resultType.size() == 0)
      return mlir::Value{}; // subroutine call
    // For now, Fortran returned values are implemented with a single MLIR
    // function return value.
    assert(call.getNumResults() == 1 &&
           "Expected exactly one result in FUNCTION call");
    return call.getResult(0);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
             &funRef) {
    llvm::SmallVector<mlir::Type, 1> resTy;
    resTy.push_back(converter.genType(TC, KIND));
    return genProcedureRef(funRef, resTy);
  }

  fir::ExtendedValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    llvm::SmallVector<mlir::Type, 1> resTy;
    if (procRef.hasAlternateReturns())
      resTy.push_back(builder.getIndexType());
    return genProcedureRef(procRef, resTy);
  }

  template <typename A>
  bool isScalar(const A &x) {
    return x.Rank() == 0;
  }
  template <typename A>
  fir::ExtendedValue asArray(const A &x) {
    auto temp = createSomeArrayTemp(converter, toEvExpr(x), symMap, stmtCtx);
    auto arrTy = fir::dyn_cast_ptrEleTy(temp.getType())
                     .template cast<fir::SequenceType>();
    auto arrShape = arrTy.getShape();
    auto shapeTy = fir::ShapeType::get(builder.getContext(), arrShape.size());
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value, 8> exprShape;
    auto loc = getLoc();
    if (arrTy.hasConstantShape()) {
      for (auto s : arrShape)
        exprShape.push_back(builder.createIntegerConstant(loc, idxTy, s));
    } else {
      for (auto s : temp.getShapeOperands())
        exprShape.push_back(builder.createConvert(loc, idxTy, s));
    }
    auto shape = builder.create<fir::ShapeOp>(loc, shapeTy, exprShape);
    mlir::Value slice;
    auto arrLd = builder.create<fir::ArrayLoadOp>(loc, arrTy, temp, shape,
                                                  slice, llvm::None);
    auto loopRes = Fortran::lower::createSomeNewArrayValue(
        converter, arrLd, {}, toEvExpr(x), symMap, stmtCtx);
    auto tempRes = temp.getResult();
    builder.create<fir::ArrayMergeStoreOp>(loc, arrLd, fir::getBase(loopRes),
                                           tempRes);
    return tempRes;
  }

  /// Lower an array value as an argument. This argument can be passed as a box
  /// value, so it may be possible to avoid making a temporary.
  template <typename A>
  fir::ExtendedValue asArrayArg(const Fortran::evaluate::Expr<A> &x) {
    return std::visit([&](const auto &e) { return asArrayArg(e, x); }, x.u);
  }
  template <typename A, typename B>
  fir::ExtendedValue asArrayArg(const Fortran::evaluate::Expr<A> &x,
                                const B &y) {
    return std::visit([&](const auto &e) { return asArrayArg(e, y); }, x.u);
  }
  template <typename A, typename B>
  fir::ExtendedValue asArrayArg(const Fortran::evaluate::Designator<A> &,
                                const B &x) {
    // Designator is being passed as an argument to a procedure. Lower the
    // expression to a boxed value.
    return Fortran::lower::createSomeArrayBox(converter, toEvExpr(x), symMap,
                                              stmtCtx);
  }
  template <typename A, typename B>
  fir::ExtendedValue asArrayArg(const A &, const B &x) {
    // If the expression to pass as an argument is not a designator, then create
    // an array temp.
    return asArray(x);
  }

  template <typename A>
  fir::ExtendedValue gen(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
    if (useBoxArg)
      return asArrayArg(x);
    return asArray(x);
  }
  template <typename A>
  fir::ExtendedValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x) ||
        inInitializer)
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    return asArray(x);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
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

  template <typename A>
  fir::ExtendedValue genref(const A &a) {
    if constexpr (inRefSet<std::decay_t<decltype(a)>>) {
      return gen(a);
    } else {
      auto exv = genval(a);
      auto valBase = fir::getBase(exv);
      // Functions are always referent.
      if (valBase.getType().template isa<mlir::FunctionType>() ||
          fir::isa_ref_type(valBase.getType()))
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
  }
  template <typename A, template <typename> typename T,
            typename B = std::decay_t<T<A>>,
            std::enable_if_t<
                std::is_same_v<B, Fortran::evaluate::Expr<A>> ||
                    std::is_same_v<B, Fortran::evaluate::Designator<A>> ||
                    std::is_same_v<B, Fortran::evaluate::FunctionRef<A>>,
                bool> = true>
  fir::ExtendedValue genref(const T<A> &x) {
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

namespace {
class ArrayExprLowering {
  struct IterationSpace {
    IterationSpace() = default;
    explicit IterationSpace(mlir::Value inArg, mlir::Value outRes,
                            llvm::ArrayRef<mlir::Value> indices)
        : inArg{inArg}, outRes{outRes}, indices{indices.begin(),
                                                indices.end()} {}

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

  private:
    mlir::Value inArg;
    mlir::Value outRes;
    llvm::SmallVector<mlir::Value, 8> indices;
  };

public:
  using ExtValue = fir::ExtendedValue;
  using IterSpace = const IterationSpace &;      // active iteration space
  using CC = std::function<ExtValue(IterSpace)>; // current continuation
  using PC =
      std::function<IterationSpace(IterSpace)>; // projection continuation

  static fir::ArrayLoadOp lowerArraySubspace(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap, /*inProjection=*/true};
    return ael.lowerArrayProjection(expr);
  }

  fir::ArrayLoadOp lowerArrayProjection(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          auto exv = f(IterationSpace{});
          if (auto *defOp = fir::getBase(exv).getDefiningOp())
            if (auto arrLd = mlir::dyn_cast<fir::ArrayLoadOp>(defOp))
              return arrLd;
          llvm::report_fatal_error("array must be loaded");
        },
        exp.u);
  }

  /// This is the entry-point into lowering an expression with rank.
  static ExtValue lowerArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      fir::ArrayLoadOp dst,
      const std::optional<Fortran::evaluate::Shape> &shape,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap, dst, shape};
    return ael.lowerArrayExpr(expr);
  }

  static ExtValue lowerAndBoxArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap};
    ael.setUseEmbox();
    return ael.boxArrayExpr(expr);
  }

  /// For an elemental array expression.
  /// 1. Lower the scalars and array loads.
  /// 2. Create the iteration space.
  /// 3. Create the element-by-element computation in the loop.
  /// 4. Return the resulting array value.
  ExtValue lowerArrayExpr(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto loc = getLoc();
          auto f = genarr(e);
          auto [iterSpace, insPt] = genIterSpace();
          auto exv = f(iterSpace);
          auto innerArg = iterSpace.innerArgument();
          auto upd = builder.create<fir::ArrayUpdateOp>(
              loc, innerArg.getType(), innerArg, fir::getBase(exv),
              iterSpace.iterVec());
          builder.create<fir::ResultOp>(loc, upd.getResult());
          builder.restoreInsertionPoint(insPt);
          return fir::substBase(exv, iterSpace.outerResult());
        },
        exp.u);
  }

  ExtValue boxArrayExpr(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          auto exv = f(IterationSpace{});
          if (auto *defOp = fir::getBase(exv).getDefiningOp())
            if (auto arrLd = mlir::dyn_cast<fir::EmboxOp>(defOp))
              return exv;
          llvm::report_fatal_error("array must be emboxed");
        },
        exp.u);
  }

  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint> genIterSpace() {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    llvm::SmallVector<mlir::Value, 8> loopUppers;

    // Convert the shape to closed interval form.
    if (destShape.has_value()) {
      // Use the shape provided.
      for (const auto &s : *destShape) {
        auto ub = builder.createConvert(
            loc, idxTy, convertOptExtentExpr(converter, stmtCtx, s));
        auto up = builder.create<mlir::SubIOp>(loc, ub, one);
        loopUppers.push_back(up);
      }
    } else {
      // Otherwise, use the array's declared shape.
      for (auto s : destination.getExtents()) {
        auto ub = builder.createConvert(loc, idxTy, s);
        auto up = builder.create<mlir::SubIOp>(loc, ub, one);
        loopUppers.push_back(up);
      }
    }

    // Iteration space is created with outermost columns, innermost rows
    fir::DoLoopOp inner;
    auto innerArg = destination.getResult();
    mlir::Value outerRes;
    llvm::SmallVector<mlir::Value, 8> ivars;
    auto insPt = builder.saveInsertionPoint();
    const auto loopDepth = loopUppers.size();
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
    return {IterationSpace{innerArg, outerRes, ivars}, insPt};
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

  template <typename A, typename = std::enable_if_t<Fortran::common::HasMember<
                            A, Fortran::evaluate::TypelessExpression>>>
  CC genarr(const A &x) {
    // Ev::ProcedureDesignator and Ev::ProcedureRef may yield a non-zero rank,
    // but treat them as scalar values.
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
  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  CC genarr(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                             TC2> &x) {
    assert(isArray(x));
    auto loc = getLoc();
    auto lf = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = fir::getBase(lf(iters));
      auto ty = converter.genType(TC1, KIND);
      return builder.createConvert(loc, ty, val);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto isImagPart = x.isImaginaryPart;
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
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
    auto lf = genarr(evEx.left());
    auto rf = genarr(evEx.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
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
  GENBIN(Add, Real, fir::AddfOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::SubIOp)
  GENBIN(Subtract, Real, fir::SubfOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::MulIOp)
  GENBIN(Multiply, Real, fir::MulfOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::SignedDivIOp)
  GENBIN(Divide, Real, fir::DivfOp)
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
  template <int KIND>
  CC genarr(const Fortran::evaluate::Concat<KIND> &x) {
    TODO("concat");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::SetLength<KIND> &x) {
    TODO("set length");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  template <typename A>
  CC genarr(const Fortran::evaluate::Constant<A> &x) {
    return [](IterSpace iters) -> ExtValue {
      TODO("constant");
      return mlir::Value{}; /* FIXME */
    };
  }

  // A vector subscript expression may be wrapped with a cast to INTEGER*8. Get
  // rid of it here so the vector can be loaded. Add it back when generating the
  // elemental evaluation (inside the loop nest).
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

  // Get the `Se::Symbol*` for the subscript expression, `x`. This symbol can be
  // used to determine the lbound, ubound of the vector.
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

  static mlir::Value getLBoundOrDefault(mlir::Location loc,
                                        const fir::ExtendedValue &exv,
                                        mlir::Value one, unsigned dim) {
    auto getLBound = [&](const fir::AbstractArrayBox &v) -> mlir::Value {
      auto &lbounds = v.getLBounds();
      if (lbounds.empty())
        return one;
      return lbounds[dim];
    };
    return exv.match(
        [&](const fir::ArrayBoxValue &v) { return getLBound(v); },
        [&](const fir::CharArrayBoxValue &v) { return getLBound(v); },
        [&](const fir::BoxValue &v) { return getLBound(v); },
        [&](auto) -> mlir::Value {
          fir::emitFatalError(loc, "expected array");
        });
  }

  /// Array reference with subscripts. Since this has rank > 0, this is a form
  /// of an array section (slice).
  ///
  /// There are two "slicing" primitives that may be applied on a dimension by
  /// dimension basis: (1) triple notation and (2) vector addressing. Since
  /// dimensions can be selectively sliced, some dimensions may contain regular
  /// scalar expressions and those dimensions do not participate in the array
  /// expression evaluation.
  CC genarr(const Fortran::evaluate::ArrayRef &x) {
    llvm::SmallVector<mlir::Value, 9> trips;
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
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
                  TODO("lbound");
                if (auto optUp = t.upper())
                  trips.push_back(fir::getBase(asScalar(*optUp)));
                else
                  TODO("ubound");
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
                  mlir::Value arrLd =
                      lowerArraySubspace(converter, symMap, stmtCtx, arrExpr);
                  auto eleTy =
                      arrLd.getType().cast<fir::SequenceType>().getEleTy();
                  auto currentPC = pc;
                  auto dim = sub.index();
                  auto lb = getLBoundOrDefault(loc, exv, one, dim);
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    auto iter = newIters.iterVec()[dim];
                    auto fetch = builder.create<fir::ArrayFetchOp>(
                        loc, eleTy, arrLd, mlir::ValueRange{iter});
                    auto cast = builder.createConvert(loc, idxTy, fetch);
                    auto val =
                        builder.create<mlir::SubIOp>(loc, idxTy, cast, lb);
                    newIters.setIndexValue(dim, val);
                    return newIters;
                  };
                  auto useInexactRange = [&]() {
                    // FIXME: Just using MAX_INT here as a fallback.
                    trips.push_back(one);
                    auto dummy = builder.createIntegerConstant(
                        loc, idxTy, std::numeric_limits<std::int64_t>::max());
                    trips.push_back(dummy);
                    trips.push_back(one);
                  };
                  if (const auto *sym = extractSubscriptSymbol(arrExpr)) {
                    auto symVal = symMap.lookupSymbol(*sym);
                    symVal.match(
                        [&](const fir::ArrayBoxValue &v) {
                          auto orig = builder.createConvert(
                              loc, idxTy,
                              v.getLBounds().empty() ? one : v.getLBounds()[0]);
                          trips.push_back(orig);
                          auto extent = builder.createConvert(
                              loc, idxTy, v.getExtents()[0]);
                          auto sum = builder.create<mlir::AddIOp>(loc, idxTy,
                                                                  orig, extent);
                          mlir::Value ubound = builder.create<mlir::SubIOp>(
                              loc, idxTy, sum, one);
                          trips.push_back(ubound);
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
                  auto lb = getLBoundOrDefault(loc, exv, one, dim);
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
    auto lf = genSlice(x.base(), trips);
    return [=](IterSpace iters) { return lf(pc(iters)); };
  }
  CC genarr(const Fortran::evaluate::NamedEntity &entity) {
    if (entity.IsSymbol())
      return genarr(entity.GetFirstSymbol());
    return genarr(entity.GetComponent());
  }
  CC genarr(const Fortran::semantics::SymbolRef &sym) {
    auto loc = getLoc();
    auto extMemref = asScalarRef(sym);
    auto memref = fir::getBase(extMemref);
    auto arrTy = fir::dyn_cast_ptrEleTy(memref.getType());
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
    if (useEmbox) {
      auto boxTy = fir::BoxType::get(reduceRank(arrTy, slice));
      mlir::Value embox = builder.create<fir::EmboxOp>(
          loc, boxTy, memref, shape, slice, /*lenParams=*/llvm::None);
      return [=](IterSpace) -> ExtValue { return embox; };
    }
    mlir::Value arrLd = builder.create<fir::ArrayLoadOp>(
        loc, arrTy, memref, shape, slice, /*lenParams=*/llvm::None);
    if (inProjection)
      return [=](IterSpace) -> ExtValue { return arrLd; };
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    return [=](IterSpace iters) -> ExtValue {
      return emboxElement(extMemref, builder.create<fir::ArrayFetchOp>(
                                         loc, eleTy, arrLd, iters.iterVec()));
    };
  }

  static ExtValue emboxElement(const ExtValue &memref, mlir::Value arrFetch) {
    return memref.match(
        [&](const fir::CharBoxValue &cb) -> ExtValue {
          return cb.clone(arrFetch);
        },
        [&](const fir::CharArrayBoxValue &bv) -> ExtValue {
          return bv.cloneElement(arrFetch);
        },
        [&](const fir::BoxValue &bv) -> ExtValue {
          return bv.cloneElement(arrFetch);
        },
        [&](const auto &) -> ExtValue { return arrFetch; });
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
    llvm::SmallVector<mlir::Value, 8> components;
    const auto &sym = x.GetFirstSymbol();
    auto recTy = converter.genType(sym);
    buildComponentsPath(components, recTy, x.base());
    auto lf = genPathSlice(sym, components);
    return [=](IterSpace iters) { return lf(iters); };
  }

  /// The `Ev::Component` structure is tailmost down to head, so the expression
  /// <code>a%b%c</code> will be presented as <code>(component (dataref
  /// (component (dataref (symbol 'a)) (symbol 'b))) (symbol 'c))</code>.
  void buildComponentsPath(llvm::SmallVectorImpl<mlir::Value> &components,
                           mlir::Type &recTy,
                           const Fortran::evaluate::DataRef &dr) {
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::evaluate::Component &c) {
                     buildComponentsPath(components, recTy, c.base());
                     auto loc = getLoc();
                     auto name = toStringRef(c.GetLastSymbol().name());
                     components.push_back(
                         builder.create<fir::FieldIndexOp>(loc, name, recTy));
                     recTy = recTy.cast<fir::RecordType>().getType(name);
                   },
                   [&](const Fortran::semantics::SymbolRef &y) {
                     // base symbol to be sliced
                     assert(dr.Rank() > 0);
                   },
                   [&](const Fortran::evaluate::ArrayRef &r) {
                     // Must be scalar per C919 and C925
                     TODO("field name and array arguments");
                   },
                   [&](const Fortran::evaluate::CoarrayRef &r) {
                     // Must be scalar per C919 and C925
                     TODO("field name and coarray arguments");
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
    auto lf = genPathSlice(x.complex(), {offset});
    return [=](IterSpace iters) { return lf(iters); };
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

  CC genarr(const Fortran::evaluate::CoarrayRef &) { TODO("coarray ref"); }
  CC genarr(const Fortran::evaluate::Substring &) { TODO("substring"); }

  template <typename A>
  CC genarr(const Fortran::evaluate::FunctionRef<A> &x) {
    return [](IterSpace iters) -> ExtValue {
      TODO("function ref");
      return mlir::Value{}; /* FIXME */
    };
  }
  template <typename A>
  CC genarr(const Fortran::evaluate::ArrayConstructor<A> &x) {
    return [](IterSpace iters) -> ExtValue {
      TODO("array ctor");
      return mlir::Value{}; /* FIXME */
    };
  }
  CC genarr(const Fortran::evaluate::ImpliedDoIndex &x) {
    TODO("implied do index");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::TypeParamInquiry &x) {
    TODO("type parameter inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::DescriptorInquiry &x) {
    TODO("descriptor inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::StructureConstructor &x) {
    TODO("structure constructor");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }

  //===--------------------------------------------------------------------===//
  // LOCICAL operators (.NOT., .AND., .EQV., etc.)
  //===--------------------------------------------------------------------===//

  template <int KIND>
  CC genarr(const Fortran::evaluate::Not<KIND> &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto truth = builder.createBool(loc, true);
    return [=](IterSpace iters) -> ExtValue {
      auto logical = fir::getBase(lf(iters));
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
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::genCharCompare(converter, loc, pred, lhs, rhs);
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
                             Fortran::lower::SymMap &symMap, bool projection,
                             fir::ArrayLoadOp dst = {})
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, destination{dst}, inProjection{
                                                                projection} {}

  mlir::Location getLoc() { return converter.getCurrentLocation(); }
  void setUseEmbox(bool embox = true) { useEmbox = embox; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  fir::ArrayLoadOp destination;
  std::optional<Fortran::evaluate::Shape> destShape;
  llvm::SmallVector<mlir::Value, 8> sliceTriple;
  llvm::SmallVector<mlir::Value, 8> slicePath;
  bool useEmbox{false};
  bool inSlice{false};
  bool inProjection{false};
};
} // namespace

/// Given an array expression, `x`, the shape of the expression might be a
/// runtime value. In that case, drill down into the expression and find the
/// subexpression for which that dynamic shape can be found.
/// Return a vector of the ssa-values that describe the shape.
template <typename A>
static std::vector<mlir::Value>
scavengeShapeFromSomeExpr(Fortran::lower::AbstractConverter &converter,
                          const A &x,
                          Fortran::lower::StatementContext &stmtCtx) {
  auto optShape = Fortran::evaluate::GetShape(converter.getFoldingContext(), x);
  auto loc = converter.getCurrentLocation();
  if (optShape.has_value()) {
    std::vector<mlir::Value> extents;
    for (const auto &se : *optShape) {
      auto ext = convertOptExtentExpr(converter, stmtCtx, se);
      auto &builder = converter.getFirOpBuilder();
      auto idxTy = builder.getIndexType();
      extents.push_back(builder.createConvert(loc, idxTy, ext));
    }
    return extents;
  }
  fir::emitFatalError(loc, "shape analysis failed");
}

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
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.genExtAddr(expr);
}

fir::ArrayLoadOp Fortran::lower::createSomeArraySubspace(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "onto array: ") << '\n');
  return ArrayExprLowering::lowerArraySubspace(converter, symMap, stmtCtx,
                                               expr);
}

fir::AllocMemOp Fortran::lower::createSomeArrayTemp(
    AbstractConverter &converter,
    const evaluate::Expr<evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  auto *bldr = &converter.getFirOpBuilder();
  auto ty = converter.genType(expr);
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "temp array: ") << '\n');
  auto seqTy = ty.dyn_cast<fir::SequenceType>();
  assert(seqTy && "must be an array");
  auto loc = converter.getCurrentLocation();
  if (ty.cast<fir::SequenceType>().hasConstantShape()) {
    auto result = bldr->create<fir::AllocMemOp>(loc, ty);
    auto res = result.getResult();
    stmtCtx.attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, res); });
    return result;
  }
  auto result = bldr->create<fir::AllocMemOp>(
      loc, ty, ".array.expr", llvm::None,
      scavengeShapeFromSomeExpr(converter, expr, stmtCtx));
  auto res = result.getResult();
  stmtCtx.attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, res); });
  return result;
}

fir::ExtendedValue Fortran::lower::createSomeNewArrayValue(
    Fortran::lower::AbstractConverter &converter, fir::ArrayLoadOp dst,
    const std::optional<Fortran::evaluate::Shape> &shape,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  return ArrayExprLowering::lowerArrayExpression(converter, symMap, stmtCtx,
                                                 dst, shape, expr);
}

fir::ExtendedValue Fortran::lower::createSomeArrayBox(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "box designator: ") << '\n');
  return ArrayExprLowering::lowerAndBoxArrayExpression(converter, symMap,
                                                       stmtCtx, expr);
}

fir::ExtendedValue Fortran::lower::createStringLiteral(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    llvm::StringRef str, uint64_t len) {
  assert(str.size() == len);
  Fortran::lower::SymMap dummySymbolMap;
  Fortran::lower::StatementContext dummyStmtCtx;
  LLVM_DEBUG(llvm::dbgs() << "string-lit: \"" << str << "\"\n");
  return ScalarExprLowering{loc, converter, dummySymbolMap, dummyStmtCtx}
      .genStringLit(str, len);
}

fir::MutableBoxValue Fortran::lower::createSomeMutableBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  // MutableBox lowering StatementContext does not need to be propagated
  // to the caller because the result value is a variable, not a temporary
  // expression. The StatementContext clean-up can occur before using the
  // resulting MutableBoxValue.
  Fortran::lower::StatementContext dummyStmtCtx;
  return ScalarExprLowering{loc, converter, symMap, dummyStmtCtx}
      .genMutableBoxValue(expr);
}
