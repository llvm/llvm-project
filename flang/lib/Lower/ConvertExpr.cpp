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
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/CharacterRuntime.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/APFloat.h"
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

  fir::ExtendedValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  fir::ExtendedValue genStringLit(llvm::StringRef str, std::uint64_t len) {
    return genScalarLit<1>(str.str(), static_cast<int64_t>(len));
  }

  mlir::Location getLoc() { return location; }

  template <typename A>
  mlir::Value genunbox(const A &expr) {
    auto e = genval(expr);
    if (auto *r = e.getUnboxed())
      return *r;
    return {};
  }

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

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    auto type = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
    return builder.createIntegerConstant(getLoc(), type, value);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(mlir::MLIRContext *context, bool value) {
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

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::FunctionType createFunctionType() {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      auto output =
          converter.genType(Fortran::common::TypeCategory::Integer, KIND);
      llvm::SmallVector<mlir::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return mlir::FunctionType::get(inputs, output, builder.getContext());
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      auto output = Fortran::lower::convertReal(builder.getContext(), KIND);
      llvm::SmallVector<mlir::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return mlir::FunctionType::get(inputs, output, builder.getContext());
    } else {
      llvm_unreachable("this category is not implemented");
    }
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

  fir::ExtendedValue
  genAllocatableOrPointerUnbox(Fortran::semantics::SymbolRef sym) {
    auto boxAddr = symMap.lookupSymbol(sym).getAddr();
    if (Fortran::semantics::IsAssumedRankArray(sym))
      TODO("Assumed rank allocatables or pointers");
    if (Fortran::semantics::IsPointer(sym))
      TODO("pointer"); // deal with non contiguity;
    auto rank = sym->Rank();
    // TODO: only intrinsic types other than CHARACTER
    if (!boxAddr)
      TODO("Allocatable type not lowered yet");
    auto boxType = fir::dyn_cast_ptrEleTy(boxAddr.getType());
    if (!boxType || !boxType.isa<fir::BoxType>())
      llvm_unreachable("bad allocatable or pointer type in symbol map");

    auto varAddrType = boxType.cast<fir::BoxType>().getEleTy();
    auto loc = getLoc();

    auto box = builder.create<fir::LoadOp>(loc, boxAddr);
    auto addr = builder.create<fir::BoxAddrOp>(loc, varAddrType, box);
    if (rank == 0)
      return addr;
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value, 4> lbounds;
    llvm::SmallVector<mlir::Value, 4> extents;
    for (decltype(rank) dim = 0; dim < rank; ++dim) {
      auto dimVal = builder.createIntegerConstant(loc, idxTy, dim);
      auto dimInfo =
          builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dimVal);
      lbounds.push_back(dimInfo.getResult(0));
      extents.push_back(dimInfo.getResult(1));
    }
    return fir::ArrayBoxValue{addr, extents, lbounds};
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  fir::ExtendedValue gen(Fortran::semantics::SymbolRef sym) {
    if (Fortran::semantics::IsAllocatableOrPointer(sym))
      return genAllocatableOrPointerUnbox(sym);
    if (auto val = symMap.lookupSymbol(sym))
      return val.toExtendedValue();
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
    TODO("");
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
    TODO("");
  }
  fir::ExtendedValue genval(const Fortran::evaluate::ImpliedDoIndex &) {
    TODO("");
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
    llvm_unreachable("bad descriptor inquiry");
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
    auto lhs = genunbox(part.left());
    assert(lhs && "unexpected array expression complex component");
    return extractComplexPart(lhs, part.isImaginaryPart);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::Negate<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto input = genunbox(op.left());
    if (!input)
      TODO("Array expression negation");
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      // Currently no Standard/FIR op for integer negation.
      auto zero = genIntegerConstant<KIND>(builder.getContext(), 0);
      return builder.create<mlir::SubIOp>(getLoc(), zero, input);
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return builder.create<fir::NegfOp>(getLoc(), input);
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Complex,
                    "Expected numeric type");
      return builder.create<fir::NegcOp>(getLoc(), input);
    }
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
    if (!lhs || !rhs)
      TODO("Array expression power");
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    if (!lhs || !rhs)
      TODO("Array expression power");
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  mlir::Value createComplex(fir::KindTy kind, mlir::Value real,
                            mlir::Value imag) {
    return Fortran::lower::ComplexExprHelper{builder, getLoc()}.createComplex(
        kind, real, imag);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    if (!lhs || !rhs)
      TODO("Array expression complex ctor");
    return createComplex(KIND, lhs, rhs);
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
    llvm::report_fatal_error("TODO: character array concatenate");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    if (!lhs || !rhs)
      TODO("Array expression extremum");
    llvm::SmallVector<mlir::Value, 2> operands{lhs, rhs};
    if (op.ordering == Fortran::evaluate::Ordering::Greater)
      return Fortran::lower::genMax(builder, getLoc(), operands);
    return Fortran::lower::genMin(builder, getLoc(), operands);
  }

  template <int KIND>
  fir::ExtendedValue genval(const Fortran::evaluate::SetLength<KIND> &) {
    TODO("");
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return createCompareOp<mlir::CmpIOp>(op, translateRelational(op.opr));
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return createFltCmpOp<fir::CmpfOp>(op, translateFloatRelational(op.opr));
    } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
      return createFltCmpOp<fir::CmpcOp>(op, translateFloatRelational(op.opr));
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Character);
      return createCharCompare(op, translateRelational(op.opr));
    }
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
    if (!operand)
      TODO("Array expression conversion");
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
    auto *context = builder.getContext();
    auto logical = genunbox(op.left());
    if (!logical)
      TODO("Array expression negation");
    auto one = genBoolConstant(context, true);
    auto val = builder.createConvert(getLoc(), builder.getI1Type(), logical);
    return builder.create<mlir::XOrOp>(getLoc(), val, one);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    auto i1Type = builder.getI1Type();
    auto slhs = genunbox(op.left());
    auto srhs = genunbox(op.right());
    if (!slhs || !srhs)
      TODO("Array expression logical operation");
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
      return genBoolConstant(builder.getContext(), value.IsTrue());
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
      auto cplx = genunbox(ctor);
      assert(cplx && "boxed value not handled");
      return cplx;
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
        strAttr = mlir::StringAttr::get(value, context);
      } else {
        using ET = typename std::decay_t<decltype(value)>::value_type;
        std::int64_t size = static_cast<std::int64_t>(value.size());
        auto shape = mlir::VectorType::get(
            llvm::ArrayRef<std::int64_t>{size},
            mlir::IntegerType::get(sizeof(ET) * 8, builder.getContext()));
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
        getLoc(),
        Fortran::lower::CharacterExprHelper{builder, getLoc()}.getLengthType(),
        len);
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
  /// Helper to call the correct scalar conversion based on category.
  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genScalarLit(
      const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>> &value,
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &con) {
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      return genScalarLit<KIND>(value, con.LEN());
    } else /*constexpr*/ {
      return genScalarLit<TC, KIND>(value);
    }
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
        auto charVal = fir::getBase(
            genScalarLit<Fortran::common::TypeCategory::Character, KIND>(
                con.At(subscripts), con));
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
            fir::getBase(genScalarLit<TC, KIND>(con.At(subscripts), con));
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
    // TODO:
    // - derived type constant
    //   ?? derived type cannot match the above template, can it? looks like it
    //   would have to be Constant<SomeType<TC::Derived>> instead
    if (con.Rank() > 0)
      return genArrayLit(con);
    auto opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    return genScalarLit<TC, KIND>(opt.value(), con);
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
      llvm_unreachable("unhandled constant of unknown kind");
    }
  }

  template <typename A>
  fir::ExtendedValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    TODO("");
  }

  fir::ExtendedValue gen(const Fortran::evaluate::ComplexPart &) {
    TODO("complex part");
  }
  fir::ExtendedValue genval(const Fortran::evaluate::ComplexPart &) {
    TODO("complex part");
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
    assert(lower && "boxed value not handled");
    bounds.push_back(lower);
    if (auto upperBound = s.upper()) {
      auto upper = genunbox(*upperBound);
      assert(upper && "boxed value not handled");
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
          llvm::report_fatal_error("substring base is not a CharBox");
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
    if (!upper || !lower)
      llvm::report_fatal_error("triplet not lowered to unboxed values");
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
    llvm::SmallVector<mlir::Value, 2> coorArgs;
    auto obj = genunbox(*base);
    assert(obj && "boxed value not handled");
    auto *sym = &cmpt.GetFirstSymbol();
    auto ty = converter.genType(*sym);
    for (auto *field : list) {
      sym = &field->GetLastSymbol();
      auto name = sym->name().ToString();
      // FIXME: as we're walking the chain of field names, we need to update the
      // subtype as we drill down
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(getLoc(), name, ty));
    }
    assert(sym && "no component(s)?");
    ty = builder.getRefType(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, obj, coorArgs);
  }

  fir::ExtendedValue genval(const Fortran::evaluate::Component &cmpt) {
    auto c = gen(cmpt);
    if (auto *val = c.getUnboxed())
      return genLoad(*val);
    TODO("");
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
    auto e{shape.size() - dims};
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
    for (auto &subsc : aref.subscript()) {
      auto sv = genunbox(subsc);
      assert(sv && "boxed value not handled");
      args.push_back(sv);
    }
    auto ty = genSubType(base.getType(), args.size());
    ty = builder.getRefType(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
  }

  static bool isSlice(const Fortran::evaluate::ArrayRef &aref) {
    for (auto &sub : aref.subscript()) {
      if (std::holds_alternative<Fortran::evaluate::Triplet>(sub.u))
        return true;
    }
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
    auto one = builder.createIntegerConstant(getLoc(), idxTy, 1);
    auto zero = builder.createIntegerConstant(getLoc(), idxTy, 0);
    auto getLB = [&](const auto &arr, unsigned dim) -> mlir::Value {
      return arr.getLBounds().empty() ? one : arr.getLBounds()[dim];
    };
    auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
      mlir::Value total = zero;
      assert(arr.getExtents().size() == aref.subscript().size());
      // unsigned idx = 0;
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
    auto genArraySlice = [&](const auto &arr) -> mlir::Value {
      // FIXME: create a loop nest and copy the array slice into a temp
      // We need some context here, since we could also box as an argument
      llvm::report_fatal_error("TODO: array slice not supported");
    };
    return si.match(
        [&](const Fortran::lower::SymbolBox::FullDim &arr)
            -> fir::ExtendedValue {
          if (/*!inArrayContext() &&*/ isSlice(aref))
            return genArraySlice(arr);
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
          if (/*!inArrayContext() &&*/ isSlice(aref)) {
            TODO("");
            return mlir::Value{};
          }
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
      if (Fortran::semantics::IsAllocatableOrPointer(symbol))
        si = gen(symbol).match(
            [&](const fir::ArrayBoxValue &x) -> Fortran::lower::SymbolBox {
              return x;
            },
            [&](const auto &) -> Fortran::lower::SymbolBox {
              TODO("character and derived intrinsic allocatables");
              return {};
            });
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

    // Implicit interface implementation only
    // TODO: Explicit interface, we need to use Characterize here,
    // evaluate::IntrinsicProcTable is required to use it.
    Fortran::lower::CallerInterface caller(procRef, converter);
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

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

      auto argRef = genExtAddr(*expr);

      auto helper = Fortran::lower::CharacterExprHelper{builder, getLoc()};
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
              mlir::emitError(getLoc(), "Lowering internal error: actual "
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
          Fortran::lower::CharacterExprHelper helper{builder, getLoc()};
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
            builder.create<fir::AddrOfOp>(getLoc(), funcOpType, symbolAttr);
      } else if (callSiteType.getResults() != funcOpType.getResults()) {
        // Implicit interface result type mismatch are not standard Fortran,
        // but some compilers are not complaining about it.
        // The front-end is not protecting lowering from this currently. Support
        // this with a discouraging warning.
        mlir::emitWarning(getLoc(),
                          "return type mismatches were never standard"
                          " compliant and may lead to undefined behavior.");
        // Cast the actual function to the current caller implicit type because
        // that is the behavior we would get if we could not see the definition.
        funcPointer =
            builder.create<fir::AddrOfOp>(getLoc(), funcOpType, symbolAttr);
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
      operands.push_back(
          builder.createConvert(getLoc(), funcType, funcPointer));

    // Deal with potential mismatches in arguments types. Passing an array to
    // a scalar argument should for instance be tolerated here.
    for (auto [fst, snd] :
         llvm::zip(caller.getInputs(), funcType.getInputs())) {
      auto cast = builder.convertWithSemantics(getLoc(), snd, fst);
      operands.push_back(cast);
    }

    auto call = builder.create<fir::CallOp>(getLoc(), funcType.getResults(),
                                            funcSymbolAttr, operands);
    // Handle case where result was passed as argument
    if (caller.getPassedResult()) {
      return resRef.getValue();
    }
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
    auto temp = createSomeArrayTemp(
        converter, Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x)),
        symMap, stmtCtx);
    auto arrTy = fir::dyn_cast_ptrEleTy(temp.getType())
                     .template cast<fir::SequenceType>();
    auto arrShape = arrTy.getShape();
    auto shapeTy = fir::ShapeType::get(builder.getContext(), arrShape.size());
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value, 8> exprShape;
    if (arrTy.hasConstantShape()) {
      for (auto s : arrShape)
        exprShape.push_back(builder.createIntegerConstant(getLoc(), idxTy, s));
    } else {
      for (auto s : temp.getShapeOperands())
        exprShape.push_back(builder.createConvert(getLoc(), idxTy, s));
    }
    auto shape = builder.create<fir::ShapeOp>(getLoc(), shapeTy, exprShape);
    mlir::Value slice;
    auto arrLd = builder.create<fir::ArrayLoadOp>(getLoc(), arrTy, temp, shape,
                                                  slice, llvm::None);
    auto loopRes = Fortran::lower::createSomeNewArrayValue(
        converter, arrLd,
        Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x)), symMap,
        stmtCtx);
    auto tempRes = temp.getResult();
    builder.create<fir::ArrayMergeStoreOp>(getLoc(), arrLd,
                                           fir::getBase(loopRes), tempRes);
    return tempRes;
  }

  template <typename A>
  fir::ExtendedValue gen(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
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

  private:
    mlir::Value inArg;
    mlir::Value outRes;
    llvm::SmallVector<mlir::Value, 8> indices;
  };

public:
  using ExtValue = fir::ExtendedValue;
  using IterSpace = const IterationSpace &;      // active iteration space
  using CC = std::function<ExtValue(IterSpace)>; // current continuation

  static fir::ArrayLoadOp lowerArraySubspace(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap};
    ael.setInProjection();
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
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap, dst};
    return ael.lowerArrayExpr(expr);
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
          return iterSpace.outerResult();
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
    for (auto s : destination.getExtents()) {
      auto ub = builder.createConvert(loc, idxTy, s);
      auto up = builder.create<mlir::SubIOp>(loc, ub, one);
      loopUppers.push_back(up);
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

  // An expression with non-zero rank is an array expression.
  template <typename A>
  static bool isArray(const A &x) {
    return x.Rank() != 0;
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
  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  CC genarr(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                             TC2> &x) {
    assert(isArray(x));
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = f(iters);
      auto ty = converter.genType(TC1, KIND);
      return builder.createConvert(getLoc(), ty, fir::getBase(val));
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = f(iters);
      return mlir::Value{}; /* FIXME */
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Parentheses<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = f(iters);
      auto base = fir::getBase(val);
      auto newBase =
          builder.create<fir::NoReassocOp>(getLoc(), base.getType(), base);
      return fir::substBase(val, newBase);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = fir::getBase(f(iters));
      auto ty = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
      auto zero = builder.createIntegerConstant(getLoc(), ty, 0);
      return builder.create<mlir::SubIOp>(getLoc(), zero, val);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = fir::getBase(f(iters));
      return builder.create<fir::NegfOp>(getLoc(), val);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = fir::getBase(f(iters));
      return builder.create<fir::NegcOp>(getLoc(), val);
    };
  }

  //===--------------------------------------------------------------------===//
  // Binary elemental ops
  //===--------------------------------------------------------------------===//

  template <typename OP, typename A>
  CC createBinaryOp(const A &evEx) {
    auto lf = genarr(evEx.left());
    auto rf = genarr(evEx.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
      auto right = fir::getBase(rf(iters));
      return builder.create<OP>(getLoc(), left, right);
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
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
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
      return mlir::Value{}; /* FIXME */
    };
  }
  CC genarr(const Fortran::evaluate::ArrayRef &arr) {
    return [](IterSpace iters) -> ExtValue {
      TODO("array ref");
      return mlir::Value{};
    };
  }
  CC genarr(const Fortran::semantics::SymbolRef &sym) {
    auto extMemref = symMap.lookupSymbol(sym).toExtendedValue();
    auto memref = fir::getBase(extMemref);
    auto arrTy = fir::dyn_cast_ptrEleTy(memref.getType());
    auto loc = getLoc();
    assert(arrTy.isa<fir::SequenceType>());
    auto shape = builder.createShape(loc, extMemref);
    mlir::Value slice; // not sliced
    mlir::Value arrLd = builder.create<fir::ArrayLoadOp>(
        loc, arrTy, memref, shape, slice, llvm::None);
    if (inProjection)
      return [=](IterSpace) -> ExtValue { return arrLd; };
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<fir::ArrayFetchOp>(loc, eleTy, arrLd,
                                               iters.iterVec());
    };
  }
  CC genarr(const Fortran::evaluate::Component &) { TODO("component"); }
  CC genarr(const Fortran::evaluate::CoarrayRef &) { TODO("coarray ref"); }
  CC genarr(const Fortran::evaluate::Substring &) { TODO("substring"); }
  CC genarr(const Fortran::evaluate::ComplexPart &) { TODO("complex part"); }
  template <typename A>
  CC genarr(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genarr(x); }, des.u);
  }
  template <typename A>
  CC genarr(const Fortran::evaluate::FunctionRef<A> &x) {
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
    };
  }
  template <typename A>
  CC genarr(const Fortran::evaluate::ArrayConstructor<A> &x) {
    return [](IterSpace iters) -> ExtValue {
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
  template <int KIND>
  CC genarr(const Fortran::evaluate::Not<KIND> &x) {
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::LogicalOperation<KIND> &x) {
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
    };
  }
  CC genarr(
      const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &x) {
    return [](IterSpace iters) -> ExtValue {
      return mlir::Value{}; /* FIXME */
    };
  }

private:
  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             fir::ArrayLoadOp dst = {})
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, destination{dst} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap, bool projection,
                             fir::ArrayLoadOp dst = {})
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, destination{dst}, inProjection{
                                                                projection} {}

  mlir::Location getLoc() { return converter.getCurrentLocation(); }
  void setInProjection(bool on = true) { inProjection = on; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  fir::ArrayLoadOp destination;
  bool inProjection{false};
};
} // namespace

namespace {
// FIXME: The `AnyTraverse` pattern doesn't short-circuit its visitation. That
// is, the entire `Ev::Expr` tree is visited, but only the first non-`None`
// result wins. The root cause is the use of `Combine()` in the `Traverse`
// visitor pattern. Furthermore, the `operator()` methods are all `const`, a
// usage obstacle such that any visitor class has to use `mutable` internal
// state to be able to carry its own context.
struct IntraexprShapeAnalyzer
    : public Fortran::evaluate::AnyTraverse<IntraexprShapeAnalyzer,
                                            Fortran::lower::SymbolBox> {
  using Result = Fortran::lower::SymbolBox;
  using Base = Fortran::evaluate::AnyTraverse<IntraexprShapeAnalyzer, Result>;

  IntraexprShapeAnalyzer(Fortran::lower::SymMap &symMap)
      : Base(*this), symMap(symMap), rankedObj(false) {}

  using Base::operator();

  template <typename A>
  Result visit(const A &x) const {
    auto saved = rankedObj;
    rankedObj = true;
    auto result = Base::operator()(x);
    rankedObj = saved;
    return result;
  }

  Result operator()(const Fortran::evaluate::Component &x) const {
    if (x.Rank())
      TODO("derived type component");
    return {};
  }
  Result operator()(const Fortran::evaluate::NamedEntity &x) const {
    if (x.Rank())
      return visit(x);
    return {};
  }
  Result operator()(const Fortran::evaluate::ArrayRef &x) const {
    if (x.Rank())
      TODO("array ref");
    return {};
  }
  Result operator()(const Fortran::evaluate::CoarrayRef &x) const {
    if (x.Rank())
      TODO("coarray ref");
    return {};
  }
  Result operator()(const Fortran::evaluate::DataRef &x) const {
    if (x.Rank())
      return visit(x);
    return {};
  }
  Result operator()(const Fortran::evaluate::ProcedureRef &x) const {
    // FIXME: just want the return value shape of a function result here
    if (x.Rank()) {
      TODO("function result");
      return visit(x.proc());
    }
    return {};
  }
  template <typename A>
  Result operator()(const Fortran::evaluate::Designator<A> &x) const {
    if (x.Rank())
      return visit(x);
    return {};
  }
  template <typename A>
  Result operator()(const Fortran::evaluate::Variable<A> &x) const {
    if (x.Rank())
      return visit(x);
    return {};
  }
  Result operator()(const Fortran::semantics::Symbol &s) const {
    if (rankedObj) {
      LLVM_DEBUG(llvm::dbgs() << "intraexpr scope symbol: "
                              << toStringRef(s.name()) << '\n');
      return symMap.lookupSymbol(s);
    }
    return {};
  }

  Fortran::lower::SymMap &symMap;
  mutable bool rankedObj;
};
} // namespace

// Given an array expression, the shape of the expression might be a runtime
// value. In that case, drill down into the expression and find the
// subexpression for which that dynamic shape can be found.
// Return a vector of the ssa-values that describe the shape.
static std::vector<mlir::Value> scavengeShapeFromSomeExpr(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &x,
    Fortran::lower::SymMap &symMap) {
  if (auto sBox = IntraexprShapeAnalyzer(symMap)(x)) {
    std::vector<mlir::Value> extents;
    auto f = [&](const fir::AbstractArrayBox &box) {
      const auto &exts = box.getExtents();
      extents.insert(extents.end(), exts.begin(), exts.end());
    };
    sBox.toExtendedValue().match(
        [&](const fir::ArrayBoxValue &box) { f(box); },
        [&](const fir::CharArrayBoxValue &box) { f(box); },
        [&](const fir::BoxValue &box) { f(box); },
        [](auto) { llvm::report_fatal_error("not an array"); });
    return extents;
  }
  llvm::report_fatal_error("cannot find a shape");
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
  auto result =
      bldr->create<fir::AllocMemOp>(loc, ty, ".array.expr", llvm::None,
                                    scavengeShapeFromSomeExpr(expr, symMap));
  auto res = result.getResult();
  stmtCtx.attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, res); });
  return result;
}

fir::ExtendedValue Fortran::lower::createSomeNewArrayValue(
    Fortran::lower::AbstractConverter &converter, fir::ArrayLoadOp dst,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  return ArrayExprLowering::lowerArrayExpression(converter, symMap, stmtCtx,
                                                 dst, expr);
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
