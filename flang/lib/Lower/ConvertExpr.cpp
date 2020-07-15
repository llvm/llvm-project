//===-- ConvertExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertExpr.h"
#include "SymbolMap.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/CharacterRuntime.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Runtime.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define TODO() llvm_unreachable("not yet implemented")

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ExprLowering {
public:
  explicit ExprLowering(mlir::Location loc,
                        Fortran::lower::AbstractConverter &converter,
                        Fortran::lower::SymMap &map,
                        const Fortran::lower::ExpressionContext &context)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, symMap{map}, exprCtx{context} {}

  /// Lower the expression `expr` into MLIR standard dialect
  mlir::Value genAddr(const Fortran::lower::SomeExpr &expr) {
    return fir::getBase(gen(expr));
  }

  fir::ExtendedValue genExtAddr(const Fortran::lower::SomeExpr &expr) {
    return gen(expr);
  }

  mlir::Value genValue(const Fortran::lower::SomeExpr &expr) {
    return fir::getBase(genval(expr));
  }

  fir::ExtendedValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  fir::ExtendedValue genStringLit(llvm::StringRef str, std::uint64_t len) {
    return genScalarLit<1>(str.str(), static_cast<int64_t>(len));
  }

private:
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  Fortran::lower::SymMap &symMap;
  const Fortran::lower::ExpressionContext &exprCtx;

  mlir::Location getLoc() { return location; }

  template <typename A>
  mlir::Value genunbox(const A &expr) {
    auto e = genval(expr);
    if (auto *r = e.getUnboxed())
      return *r;
    llvm::report_fatal_error("value is not unboxed");
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
    auto attr = builder.getIntegerAttr(type, value);
    return builder.create<mlir::ConstantOp>(getLoc(), type, attr);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(mlir::MLIRContext *context, bool value) {
    auto i1Type = builder.getI1Type();
    auto attr = builder.getIntegerAttr(i1Type, value ? 1 : 0);
    return builder.create<mlir::ConstantOp>(getLoc(), i1Type, attr).getResult();
  }

  template <int KIND>
  mlir::Value genRealConstant(mlir::MLIRContext *context,
                              const llvm::APFloat &value) {
    auto fltTy = Fortran::lower::convertReal(context, KIND);
    auto attr = builder.getFloatAttr(fltTy, value);
    auto res = builder.create<mlir::ConstantOp>(getLoc(), fltTy, attr);
    return res.getResult();
  }

  mlir::Type getSomeKindInteger() { return builder.getIndexType(); }

  template <typename OpTy>
  mlir::Value createBinaryOp(const fir::ExtendedValue &left,
                             const fir::ExtendedValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed()) {
        assert(lhs && rhs && "argument did not lower");
        return builder.create<OpTy>(getLoc(), *lhs, *rhs);
      }
    // binary ops can appear in array contexts
    TODO();
  }
  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    return createBinaryOp<OpTy>(genval(ex.left()), genval(ex.right()));
  }

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
    TODO();
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
    TODO();
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
    if (auto *lhs = left.getUnboxed()) {
      if (auto *rhs = right.getUnboxed()) {
        return Fortran::lower::genBoxCharCompare(converter, getLoc(), pred,
                                                 *lhs, *rhs);
      } else if (auto *rhs = right.getCharBox()) {
        return Fortran::lower::genBoxCharCompare(converter, getLoc(), pred,
                                                 *lhs, rhs->getBuffer());
      }
    }
    if (auto *lhs = left.getCharBox()) {
      if (auto *rhs = right.getCharBox()) {
        // FIXME: this should be passing the CharBoxValues and not just a buffer
        // addresses
        return Fortran::lower::genBoxCharCompare(
            converter, getLoc(), pred, lhs->getBuffer(), rhs->getBuffer());
      } else if (auto *rhs = right.getUnboxed()) {
        return Fortran::lower::genBoxCharCompare(converter, getLoc(), pred,
                                                 lhs->getBuffer(), *rhs);
      }
    }

    // Error if execution reaches this point
    mlir::emitError(getLoc(), "Unhandled character comparison");
    exit(1);
  }

  template <typename A>
  mlir::Value createCharCompare(const A &ex, mlir::CmpIPredicate pred) {
    return createCharCompare(pred, genval(ex.left()), genval(ex.right()));
  }

  fir::ExtendedValue getExValue(const Fortran::lower::SymbolBox &symBox) {
    using T = fir::ExtendedValue;
    return std::visit(
        Fortran::common::visitors{
            [](const Fortran::lower::SymbolBox::Intrinsic &box) -> T {
              return box.getAddr();
            },
            [](const auto &box) -> T { return box; },
            [](const Fortran::lower::SymbolBox::None &) -> T {
              llvm_unreachable("symbol not mapped");
            }},
        symBox.box);
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  fir::ExtendedValue gen(Fortran::semantics::SymbolRef sym) {
    if (auto val = symMap.lookupSymbol(sym))
      return getExValue(val);
    llvm_unreachable("all symbols should be in the map");
    auto addr = builder.createTemporary(getLoc(), converter.genType(sym),
                                        sym->name().ToString());
    symMap.addSymbol(sym, addr);
    return addr;
  }

  mlir::Value genLoad(mlir::Value addr) {
    return builder.create<fir::LoadOp>(getLoc(), addr);
  }

  // FIXME: replace this
  mlir::Type peelType(mlir::Type ty, int count) {
    if (count > 0) {
      if (auto eleTy = fir::dyn_cast_ptrEleTy(ty))
        return peelType(eleTy, count - 1);
      if (auto seqTy = ty.dyn_cast<fir::SequenceType>())
        return peelType(seqTy.getEleTy(), count - seqTy.getDimension());
      llvm_unreachable("unhandled type");
    }
    return ty;
  }

  fir::ExtendedValue genval(Fortran::semantics::SymbolRef sym) {
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
            addr = builder.createConvert(getLoc(),
                                         builder.getRefType(resultType), addr);
        }
        return genLoad(addr);
      }
    if (inArrayContext()) {
      // FIXME: make this more robust
      auto base = fir::getBase(var);
      auto ty = builder.getRefType(
          peelType(base.getType(), exprCtx.getLoopVars().size() + 1));
      auto coor = builder.create<fir::CoordinateOp>(getLoc(), ty, base,
                                                    exprCtx.getLoopVars());
      return genLoad(coor);
    }
    return var;
  }

  fir::ExtendedValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO();
  }
  fir::ExtendedValue
  genval(const Fortran::evaluate::ProcedureDesignator &proc) {
    if (const auto *intrinsic = proc.GetSpecificIntrinsic()) {
      auto signature = Fortran::lower::translateSignature(proc, converter);
      auto symbolRefAttr =
          Fortran::lower::getUnrestrictedIntrinsicSymbolRefAttr(
              builder, getLoc(), intrinsic->name, signature);
      mlir::Value funcPtr =
          builder.create<mlir::ConstantOp>(getLoc(), signature, symbolRefAttr);
      return funcPtr;
    }
    const auto *symbol = proc.GetSymbol();
    assert(symbol && "expected symbol in ProcedureDesignator");
    if (Fortran::semantics::IsDummy(*symbol)) {
      auto val = symMap.lookupSymbol(*symbol);
      assert(val && "Dummy procedure not in symbol map");
      return val;
    }
    auto name = converter.mangleName(*symbol);
    auto func = builder.getNamedFunction(name);
    // TODO: If this is an external not called/defined in this file
    // (e.g, it is just being passed as a dummy procedure argument)
    // we need to create a funcOp for it with the interface we have.
    if (!func)
      TODO();
    mlir::Value funcPtr = builder.create<mlir::ConstantOp>(
        getLoc(), func.getType(), builder.getSymbolRefAttr(name));
    return funcPtr;
  }
  fir::ExtendedValue genval(const Fortran::evaluate::NullPointer &) {
    return builder.createNullConstant(location);
  }
  fir::ExtendedValue genval(const Fortran::evaluate::StructureConstructor &) {
    TODO();
  }
  fir::ExtendedValue genval(const Fortran::evaluate::ImpliedDoIndex &) {
    TODO();
  }

  fir::ExtendedValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    auto descRef = symMap.lookupSymbol(desc.base().GetLastSymbol());
    assert(descRef && "no mlir::Value associated to Symbol");
    auto descType = descRef.getAddr().getType();
    mlir::Value res{};
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      if (descType.isa<fir::BoxCharType>()) {
        auto lenType = Fortran::lower::CharacterExprHelper{builder, getLoc()}
                           .getLengthType();
        res = builder.create<fir::BoxCharLenOp>(getLoc(), lenType, descRef);
      } else if (descType.isa<fir::BoxType>()) {
        TODO();
      } else {
        llvm_unreachable("not a descriptor");
      }
      break;
    default:
      TODO();
    }
    return res;
  }

  template <int KIND>
  fir::ExtendedValue genval(const Fortran::evaluate::TypeParamInquiry<KIND> &) {
    TODO();
  }

  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return Fortran::lower::ComplexExprHelper{builder, getLoc()}
        .extractComplexPart(cplx, isImagPart);
  }

  template <int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    auto lhs = genunbox(part.left());
    assert(lhs && "boxed type not handled");
    return extractComplexPart(lhs, part.isImaginaryPart);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::Negate<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto input = genunbox(op.left());
    assert(input && "boxed value not handled");
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      // Currently no Standard/FIR op for integer negation.
      auto zero = genIntegerConstant<KIND>(builder.getContext(), 0);
      return builder.create<mlir::SubIOp>(getLoc(), zero, input);
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return builder.create<fir::NegfOp>(getLoc(), input);
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Complex,
                    "Expected numeric type");
      return createBinaryOp<fir::NegcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Add<Fortran::evaluate::Type<TC, KIND>> &op) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return createBinaryOp<mlir::AddIOp>(op);
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return createBinaryOp<fir::AddfOp>(op);
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Complex,
                    "Expected numeric type");
      return createBinaryOp<fir::AddcOp>(op);
    }
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Subtract<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return createBinaryOp<mlir::SubIOp>(op);
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return createBinaryOp<fir::SubfOp>(op);
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Complex,
                    "Expected numeric type");
      return createBinaryOp<fir::SubcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Multiply<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return createBinaryOp<mlir::MulIOp>(op);
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return createBinaryOp<fir::MulfOp>(op);
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Complex,
                    "Expected numeric type");
      return createBinaryOp<fir::MulcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::Divide<Fortran::evaluate::Type<TC, KIND>> &op) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return createBinaryOp<mlir::SignedDivIOp>(op);
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      return createBinaryOp<fir::DivfOp>(op);
    } else {
      static_assert(TC == Fortran::common::TypeCategory::Complex,
                    "Expected numeric type");
      return createBinaryOp<fir::DivcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    assert(lhs && rhs && "boxed value not handled");
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    assert(lhs && rhs && "boxed value not handled");
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
    assert(lhs && rhs && "boxed value not handled");
    return createComplex(KIND, lhs, rhs);
  }

  template <int KIND>
  fir::ExtendedValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    auto lhs = genval(op.left());
    auto rhs = genval(op.right());
    auto lhsBase = fir::getBase(lhs);
    auto rhsBase = fir::getBase(rhs);
    return Fortran::lower::CharacterExprHelper{builder, getLoc()}
        .createConcatenate(lhsBase, rhsBase);
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    assert(lhs && rhs && "boxed value not handled");
    llvm::SmallVector<mlir::Value, 2> operands{lhs, rhs};
    if (op.ordering == Fortran::evaluate::Ordering::Greater)
      return Fortran::lower::genMax(builder, getLoc(), operands);
    return Fortran::lower::genMin(builder, getLoc(), operands);
  }

  template <int KIND>
  fir::ExtendedValue genval(const Fortran::evaluate::SetLength<KIND> &) {
    TODO();
  }

  mlir::Value createComplexCompare(mlir::Value cplx1, mlir::Value cplx2,
                                   bool eq) {
    return Fortran::lower::ComplexExprHelper{builder, getLoc()}
        .createComplexCompare(cplx1, cplx2, eq);
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
      bool eq{op.opr == Fortran::common::RelationalOperator::EQ};
      if (!eq && op.opr != Fortran::common::RelationalOperator::NE)
        llvm_unreachable("relation undefined for complex");
      auto lhs = genunbox(op.left());
      auto rhs = genunbox(op.right());
      assert(lhs && rhs && "boxed value not handled");
      return createComplexCompare(lhs, rhs, eq);
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
    assert(operand && "boxed value not handled");
    return builder.createConvert(getLoc(), ty, operand);
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
    assert(logical && "boxed value not handled");
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
    assert(slhs && srhs && "boxed value not handled");
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
    auto type = fir::SequenceType::get(
        {len}, fir::CharacterType::get(builder.getContext(), KIND));
    auto consLit = [&]() -> fir::StringLitOp {
      auto context = builder.getContext();
      auto strAttr =
          mlir::StringAttr::get((const char *)value.c_str(), context);
      auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), context);
      mlir::NamedAttribute dataAttr(valTag, strAttr);
      auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), context);
      mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
      llvm::SmallVector<mlir::NamedAttribute, 2> attrs{dataAttr, sizeAttr};
      return builder.create<fir::StringLitOp>(
          getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
    };

    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    if (exprCtx.inInitializer())
      return consLit().getResult();

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
          });
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    auto lenp = builder.createIntegerConstant(
        getLoc(),
        Fortran::lower::CharacterExprHelper{builder, getLoc()}.getLengthType(),
        len);
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
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      fir::SequenceType::Shape shape;
      shape.push_back(con.LEN());
      shape.append(con.shape().begin(), con.shape().end());
      auto chTy =
          converter.genType(Fortran::common::TypeCategory::Character, KIND);
      auto arrayTy = fir::SequenceType::get(shape, chTy);
      auto idxTy = builder.getIndexType();
      mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
      Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
      do {
        auto constant = fir::getBase(
            genScalarLit<Fortran::common::TypeCategory::Character, KIND>(
                con.At(subscripts), con));
        for (std::int64_t i = 0, L = con.LEN(); i < L; ++i) {
          llvm::SmallVector<mlir::Value, 8> idx;
          idx.push_back(builder.createIntegerConstant(getLoc(), idxTy, i));
          auto charVal = builder.create<fir::ExtractValueOp>(getLoc(), chTy,
                                                             constant, idx);
          for (const auto &pair : llvm::zip(subscripts, con.lbounds())) {
            const auto &dim = std::get<0>(pair);
            const auto &lb = std::get<1>(pair);
            idx.push_back(
                builder.createIntegerConstant(getLoc(), idxTy, dim - lb));
          }
          array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                     charVal, idx);
        }
      } while (con.IncrementSubscripts(subscripts));
      // FIXME: return an ArrayBoxValue
      return array;
    } else {
      // Convert Ev::ConstantSubs to SequenceType::Shape
      fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
      auto eleTy = converter.genType(TC, KIND);
      auto arrayTy = fir::SequenceType::get(shape, eleTy);
      auto idxTy = builder.getIndexType();
      mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
      Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
      do {
        auto constant =
            fir::getBase(genScalarLit<TC, KIND>(con.At(subscripts), con));
        llvm::SmallVector<mlir::Value, 8> idx;
        for (const auto &pair : llvm::zip(subscripts, con.lbounds())) {
          const auto &dim = std::get<0>(pair);
          const auto &lb = std::get<1>(pair);
          idx.push_back(
              builder.createIntegerConstant(getLoc(), idxTy, dim - lb));
        }
        auto insVal = builder.createConvert(getLoc(), eleTy, constant);
        array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                   insVal, idx);
      } while (con.IncrementSubscripts(subscripts));
      // FIXME: return an ArrayBoxValue
      return array;
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  fir::ExtendedValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    // TODO:
    // - derived type constant
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
    TODO();
  }

  fir::ExtendedValue gen(const Fortran::evaluate::ComplexPart &) { TODO(); }
  fir::ExtendedValue genval(const Fortran::evaluate::ComplexPart &) { TODO(); }

  /// Reference to a substring.
  fir::ExtendedValue gen(const Fortran::evaluate::Substring &s) {
    // Get base string
    auto baseString = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &x) { return gen(x); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &)
                -> fir::ExtendedValue { TODO(); },
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
    // FIXME: a string should be a CharBoxValue
    auto addr = fir::getBase(baseString);
    return Fortran::lower::CharacterExprHelper{builder, getLoc()}
        .createSubstring(addr, bounds);
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
    TODO();
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
    TODO();
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

  bool inArrayContext() { return exprCtx.inArrayContext(); }

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
      unsigned idx = 0;
      unsigned dim = 0;
      for (const auto &pair : llvm::zip(arr.getExtents(), aref.subscript())) {
        auto subVal = genComponent(std::get<1>(pair));
        if (auto *trip = std::get_if<fir::RangeBoxValue>(&subVal)) {
          // access A(i:j:k), decl A(m:n), iterspace (t1..)
          auto tlb = builder.createConvert(loc, idxTy, std::get<0>(*trip));
          auto dlb = builder.createConvert(loc, idxTy, getLB(arr, dim));
          auto diff = builder.create<mlir::SubIOp>(loc, tlb, dlb);
          assert(idx < exprCtx.getLoopVars().size());
          auto sum = builder.create<mlir::AddIOp>(loc, diff,
                                                  exprCtx.getLoopVars()[idx++]);
          auto del = builder.createConvert(loc, idxTy, std::get<2>(*trip));
          auto scaled = builder.create<mlir::MulIOp>(loc, del, delta);
          auto prod = builder.create<mlir::MulIOp>(loc, scaled, sum);
          total = builder.create<mlir::AddIOp>(loc, prod, total);
          if (auto ext = std::get<0>(pair))
            delta = builder.create<mlir::MulIOp>(loc, delta, ext);
        } else {
          auto *v = std::get_if<fir::ExtendedValue>(&subVal);
          assert(v);
          if (auto *sval = v->getUnboxed()) {
            auto val = builder.createConvert(loc, idxTy, *sval);
            auto lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
            auto diff = builder.create<mlir::SubIOp>(loc, val, lb);
            auto prod = builder.create<mlir::MulIOp>(loc, delta, diff);
            total = builder.create<mlir::AddIOp>(loc, prod, total);
            if (auto ext = std::get<0>(pair))
              delta = builder.create<mlir::MulIOp>(loc, delta, ext);
          } else {
            TODO();
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
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::lower::SymbolBox::FullDim &arr) {
              if (!inArrayContext() && isSlice(aref))
                return genArraySlice(arr);
              return genFullDim(arr, one);
            },
            [&](const Fortran::lower::SymbolBox::CharFullDim &arr) {
              return genFullDim(arr, arr.getLen());
            },
            [&](const Fortran::lower::SymbolBox::Derived &arr) {
              TODO();
              return mlir::Value{};
            },
            [&](const auto &) {
              TODO();
              return mlir::Value{};
            }},
        si.box);
  }

  // Return the coordinate of the array reference
  fir::ExtendedValue gen(const Fortran::evaluate::ArrayRef &aref) {
    if (aref.base().IsSymbol()) {
      auto &symbol = aref.base().GetFirstSymbol();
      auto si = symMap.lookupSymbol(symbol);
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
            TODO();
          }
        } else {
          auto *range = std::get_if<fir::RangeBoxValue>(&subBox);
          assert(range && "must be a range");
          // triple notation for slicing operation
          auto ty = builder.getIndexType();
          auto step = builder.createConvert(loc, ty, std::get<2>(*range));
          auto scale = builder.create<mlir::MulIOp>(
              loc, ty, exprCtx.getLoopVars()[i], step);
          auto off = builder.createConvert(loc, ty, std::get<0>(*range));
          args.push_back(builder.create<mlir::AddIOp>(loc, ty, off, scale));
        }
      }
      auto ty = genSubType(base.getType(), args.size());
      ty = builder.getRefType(ty);
      return builder.create<fir::CoordinateOp>(loc, ty, base, args);
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
    return genLoad(fir::getBase(gen(aref)));
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

  // call a function
  template <typename A>
  fir::ExtendedValue gen(const Fortran::evaluate::FunctionRef<A> &funRef) {
    TODO();
  }
  template <typename A>
  fir::ExtendedValue genval(const Fortran::evaluate::FunctionRef<A> &funRef) {
    TODO(); // Derived type functions (user + intrinsics)
  }

  fir::ExtendedValue
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  mlir::ArrayRef<mlir::Type> resultType) {
    if (resultType.size() != 1)
      TODO(); // Intrinsic subroutine

    llvm::SmallVector<fir::ExtendedValue, 2> operands;
    // Lower arguments
    // For now, logical arguments for intrinsic are lowered to `fir.logical`
    // so that TRANSFER can work. For some arguments, it could lead to useless
    // conversions (e.g scalar MASK of MERGE will be converted to `i1`), but
    // the generated code is at least correct. To improve this, the intrinsic
    // lowering facility should control argument lowering.
    for (const auto &arg : procRef.arguments()) {
      if (auto *expr = Fortran::evaluate::UnwrapExpr<
              Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg)) {
        operands.emplace_back(genval(*expr));
      } else {
        operands.emplace_back(mlir::Value{}); // absent optional
      }
    }
    // Let the intrinsic library lower the intrinsic procedure call
    llvm::StringRef name{intrinsic.name};
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name,
                                            resultType[0], operands);
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
    for (const auto &pair :
         llvm::zip(details.dummyArgs(), procRef.arguments())) {
      assert(std::get<0>(pair) && "alternate return in statement function");
      const auto &dummySymbol = *std::get<0>(pair);
      assert(std::get<1>(pair) && "optional argument in statement function");
      const auto *expr = std::get<1>(pair)->UnwrapExpr();
      // TODO: assumed type in statement function, that surprisingly seems
      // allowed, probably because nobody thought of restricting this usage.
      // gfortran/ifort compiles this.
      assert(expr && "assumed type used as statement function argument");
      auto argVal = genval(*expr);
      if (auto *charBox = argVal.getCharBox()) {
        symMap.addCharSymbol(dummySymbol, charBox->getBuffer(),
                             charBox->getLen());
      } else {
        // As per Fortran 2018 C1580, statement function arguments can only be
        // scalars, so just pass the base address.
        symMap.addSymbol(dummySymbol, fir::getBase(argVal));
      }
    }
    auto result = genval(details.stmtFunction().value());
    // Remove dummy local arguments from the map.
    for (const auto *dummySymbol : details.dummyArgs())
      symMap.erase(*dummySymbol);
    return result;
  }

  fir::ExtendedValue
  genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                  mlir::ArrayRef<mlir::Type> resultType) {
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType[0]);

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
        TODO(); // optional arguments
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(); // assumed type arguments

      mlir::Value argRef;
      mlir::Value argVal;
      if (const auto *argSymbol =
              Fortran::evaluate::UnwrapWholeSymbolDataRef(*expr)) {
        argVal = symMap.lookupSymbol(*argSymbol);
      } else {
        auto exv = genval(*expr);
        // FIXME: should use the box values, etc.
        argVal = fir::getBase(exv);
      }
      auto type = argVal.getType();
      if (fir::isa_passbyref_type(type) || type.isa<mlir::FunctionType>()) {
        argRef = argVal;
        argVal = {};
      }
      assert((argVal || argRef) && "needs value or address");

      // Handle cases where the argument must be passed by value
      if (arg.passBy == PassBy::Value) {
        if (!argVal)
          argVal = genLoad(argRef);
        caller.placeInput(arg, argVal);
        continue;
      }

      // From this point, arguments needs to be in memory.
      if (!argRef) {
        // expression is a value, so store it in a temporary so we can
        // pass-by-reference
        argRef = builder.createTemporary(getLoc(), argVal.getType());
        builder.create<fir::StoreOp>(getLoc(), argVal, argRef);
      }
      if (arg.passBy == PassBy::BaseAddress) {
        caller.placeInput(arg, argRef);
      } else if (arg.passBy == PassBy::BoxChar) {
        auto boxChar = argRef;
        if (!boxChar.getType().isa<fir::BoxCharType>()) {
          Fortran::lower::CharacterExprHelper helper{builder, getLoc()};
          auto ch = helper.materializeCharacter(boxChar);
          boxChar = helper.createEmboxChar(ch.first, ch.second);
        }
        caller.placeInput(arg, boxChar);
      } else if (arg.passBy == PassBy::Box) {
        TODO(); // generate emboxing if need.
      } else if (arg.passBy == PassBy::AddressAndLength) {
        Fortran::lower::CharacterExprHelper helper{builder, getLoc()};
        auto ch = helper.materializeCharacter(argRef);
        caller.placeAddressAndLengthInput(arg, ch.first, ch.second);
      } else {
        llvm_unreachable("pass by value not handled here");
      }
    }

    // Handle case where caller must pass result
    mlir::Value resRef;
    if (auto resultArg = caller.getPassedResult()) {
      if (resultArg->passBy == PassBy::AddressAndLength) {
        // allocate and pass character result
        auto len = caller.getResultLength();
        Fortran::lower::CharacterExprHelper helper{builder, getLoc()};
        resRef = helper.createCharacterTemp(resultType[0], len);
        auto ch = helper.createUnboxChar(resRef);
        caller.placeAddressAndLengthInput(*resultArg, ch.first, ch.second);
      } else {
        TODO(); // Pass descriptor
      }
    }

    mlir::Value funcPointer;
    mlir::SymbolRefAttr funcSymbolAttr;
    if (const auto *sym = caller.getIfIndirectCallSymbol()) {
      funcPointer = symMap.lookupSymbol(*sym);
      assert(funcPointer &&
             "dummy procedure or procedure pointer not in symbol map");
    } else {
      funcSymbolAttr = builder.getSymbolRefAttr(caller.getMangledName());
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
    // In older Fortran, procedure argument types are inferenced. Deal with
    // the potential mismatches by adding casts to the arguments when the
    // inferenced types do not match exactly.
    for (const auto &op : llvm::zip(caller.getInputs(), funcType.getInputs())) {
      auto cast = builder.convertWithSemantics(getLoc(), std::get<1>(op),
                                               std::get<0>(op));
      operands.push_back(cast);
    }

    auto call = builder.create<fir::CallOp>(getLoc(), caller.getResultType(),
                                            funcSymbolAttr, operands);
    // Handle case where result was passed as argument
    if (caller.getPassedResult())
      return resRef;
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
  fir::ExtendedValue gen(const Fortran::evaluate::Expr<A> &exp) {
    return std::visit([&](const auto &e) { return genref(e); }, exp.u);
  }
  template <typename A>
  fir::ExtendedValue genval(const Fortran::evaluate::Expr<A> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
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
  fir::ExtendedValue genref(const Fortran::evaluate::Designator<A> &x) {
    return gen(x);
  }
  template <typename A>
  fir::ExtendedValue genref(const Fortran::evaluate::FunctionRef<A> &x) {
    return gen(x);
  }
  template <typename A>
  fir::ExtendedValue genref(const Fortran::evaluate::Expr<A> &x) {
    return gen(x);
  }
  template <typename A>
  fir::ExtendedValue genref(const A &a) {
    if constexpr (inRefSet<std::decay_t<decltype(a)>>) {
      return gen(a);
    } else {
      llvm_unreachable("expression error");
    }
  }

  std::string
  applyNameMangling(const Fortran::evaluate::ProcedureDesignator &proc) {
    if (const auto *symbol = proc.GetSymbol())
      return converter.mangleName(*symbol);
    // Do not mangle intrinsic for now
    assert(proc.GetSpecificIntrinsic() &&
           "expected intrinsic procedure in designator");
    return proc.GetName();
  }
};

} // namespace

mlir::Value Fortran::lower::createSomeExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  Fortran::lower::ExpressionContext unused;
  return ExprLowering{loc, converter, symMap, unused}.genValue(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap,
    const Fortran::lower::ExpressionContext &context) {
  return ExprLowering{loc, converter, symMap, context}.genExtValue(expr);
}

mlir::Value Fortran::lower::createSomeAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  Fortran::lower::ExpressionContext unused;
  return ExprLowering{loc, converter, symMap, unused}.genAddr(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap,
    const Fortran::lower::ExpressionContext &context) {
  return ExprLowering{loc, converter, symMap, context}.genExtAddr(expr);
}

fir::ExtendedValue Fortran::lower::createStringLiteral(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    llvm::StringRef str, uint64_t len) {
  Fortran::lower::SymMap unused1;
  Fortran::lower::ExpressionContext unused2;
  return ExprLowering{loc, converter, unused1, unused2}.genStringLit(str, len);
}

//===----------------------------------------------------------------------===//
// Support functions (implemented here for now)
//===----------------------------------------------------------------------===//

mlir::Value fir::getBase(const fir::ExtendedValue &ex) {
  return std::visit(Fortran::common::visitors{
                        [](const fir::UnboxedValue &x) { return x; },
                        [](const auto &x) { return x.getAddr(); },
                    },
                    ex.box);
}

fir::ExtendedValue fir::substBase(const fir::ExtendedValue &ex,
                                  mlir::Value base) {
  return std::visit(
      Fortran::common::visitors{
          [&](const fir::UnboxedValue &x) { return fir::ExtendedValue(base); },
          [&](const auto &x) { return fir::ExtendedValue(x.clone(base)); },
      },
      ex.box);
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::CharBoxValue &box) {
  os << "boxchar { addr: " << box.getAddr() << ", len: " << box.getLen()
     << " }";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ArrayBoxValue &box) {
  os << "boxarray { addr: " << box.getAddr();
  if (box.getLBounds().size()) {
    os << ", lbounds: [";
    llvm::interleaveComma(box.getLBounds(), os);
    os << "]";
  } else {
    os << ", lbounds: all-ones";
  }
  os << ", shape: [";
  llvm::interleaveComma(box.getExtents(), os);
  os << "]}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::CharArrayBoxValue &box) {
  os << "boxchararray { addr: " << box.getAddr() << ", len : " << box.getLen();
  if (box.getLBounds().size()) {
    os << ", lbounds: [";
    llvm::interleaveComma(box.getLBounds(), os);
    os << "]";
  } else {
    os << " lbounds: all-ones";
  }
  os << ", shape: [";
  llvm::interleaveComma(box.getExtents(), os);
  os << "]}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::BoxValue &box) {
  os << "box { addr: " << box.getAddr();
  if (box.getLen())
    os << ", size: " << box.getLen();
  if (box.params.size()) {
    os << ", type params: [";
    llvm::interleaveComma(box.params, os);
    os << "]";
  }
  if (box.getLBounds().size()) {
    os << ", lbounds: [";
    llvm::interleaveComma(box.getLBounds(), os);
    os << "]";
  }
  if (box.getExtents().size()) {
    os << ", shape: [";
    llvm::interleaveComma(box.getExtents(), os);
    os << "]";
  }
  os << "}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ProcBoxValue &box) {
  os << "boxproc: { addr: " << box.getAddr() << ", context: " << box.hostContext
     << "}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ExtendedValue &ex) {
  std::visit([&](const auto &value) { os << value; }, ex.box);
  return os;
}

void Fortran::lower::SymMap::dump() const {
  auto &os = llvm::errs();
  for (auto iter : symbolMap) {
    os << "symbol [" << *iter.first << "] ->\n\t";
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::lower::SymbolBox::None &box) {
                     os << "** symbol not properly mapped **\n";
                   },
                   [&](const Fortran::lower::SymbolBox::Intrinsic &val) {
                     os << val.getAddr() << '\n';
                   },
                   [&](const auto &box) { os << box << '\n'; }},
               iter.second.box);
  }
}
