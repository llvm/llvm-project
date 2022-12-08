//===-- ConvertExprToHLFIR.cpp --------------------------------------------===//
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

#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Evaluate/shape.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertConstant.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

namespace {

/// Lower Designators to HLFIR.
class HlfirDesignatorBuilder {
public:
  HlfirDesignatorBuilder(mlir::Location loc,
                         Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::SymMap &symMap,
                         Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, symMap{symMap}, stmtCtx{stmtCtx}, loc{loc} {}

  // Character designators variant contains substrings
  using CharacterDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Character, 1>>::u);
  hlfir::EntityWithAttributes
  gen(const CharacterDesignators &designatorVariant) {
    return std::visit(
        [&](const auto &x) -> hlfir::EntityWithAttributes { return gen(x); },
        designatorVariant);
  }
  // Character designators variant contains complex parts
  using RealDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Real, 4>>::u);
  hlfir::EntityWithAttributes gen(const RealDesignators &designatorVariant) {
    return std::visit(
        [&](const auto &x) -> hlfir::EntityWithAttributes { return gen(x); },
        designatorVariant);
  }
  // All other designators are similar
  using OtherDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Integer, 4>>::u);
  hlfir::EntityWithAttributes gen(const OtherDesignators &designatorVariant) {
    return std::visit(
        [&](const auto &x) -> hlfir::EntityWithAttributes { return gen(x); },
        designatorVariant);
  }

private:
  /// Struct that is filled while visiting a part-ref (in the "visit" member
  /// function) before the top level "gen" generates an hlfir.declare for the
  /// part ref. It contains the lowered pieces of the part-ref that will
  /// become the operands of an hlfir.declare.
  struct PartInfo {
    fir::FortranVariableOpInterface base;
    llvm::SmallVector<hlfir::DesignateOp::Subscript, 8> subscripts;
    mlir::Value resultShape;
    llvm::SmallVector<mlir::Value> typeParams;
  };

  /// Generate an hlfir.declare for a part-ref given a filled PartInfo and the
  /// FIR type for this part-ref.
  fir::FortranVariableOpInterface genDeclare(mlir::Type resultValueType,
                                             PartInfo &partInfo) {
    // Compute hlfir.declare result type.
    // TODO: ensure polymorphic aspect of base of component  will be
    // preserved, as well as pointer/allocatable component aspects.
    mlir::Type resultType;
    /// Array sections may be non contiguous, so the output must be a box even
    /// when the extents are static. This can be refined later for cases where
    /// the output is know to be simply contiguous and that do not have lower
    /// bounds.
    auto charType = resultValueType.dyn_cast<fir::CharacterType>();
    if (charType && charType.hasDynamicLen())
      resultType =
          fir::BoxCharType::get(charType.getContext(), charType.getFKind());
    else if (resultValueType.isa<fir::SequenceType>() ||
             fir::hasDynamicSize(resultValueType))
      resultType = fir::BoxType::get(resultValueType);
    else
      resultType = fir::ReferenceType::get(resultValueType);

    llvm::Optional<bool> complexPart;
    llvm::SmallVector<mlir::Value> substring;
    auto designate = getBuilder().create<hlfir::DesignateOp>(
        getLoc(), resultType, partInfo.base.getBase(), "",
        /*componentShape=*/mlir::Value{}, partInfo.subscripts, substring,
        complexPart, partInfo.resultShape, partInfo.typeParams);
    return mlir::cast<fir::FortranVariableOpInterface>(
        designate.getOperation());
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::SymbolRef &symbolRef) {
    if (llvm::Optional<fir::FortranVariableOpInterface> varDef =
            getSymMap().lookupVariableDefinition(symbolRef))
      return *varDef;
    TODO(getLoc(), "lowering symbol to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Component &component) {
    TODO(getLoc(), "lowering component to HLFIR");
  }

  hlfir::EntityWithAttributes gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    PartInfo partInfo;
    mlir::Type resultType = visit(arrayRef, partInfo);
    return genDeclare(resultType, partInfo);
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::CoarrayRef &coarrayRef) {
    TODO(getLoc(), "lowering CoarrayRef to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ComplexPart &complexPart) {
    TODO(getLoc(), "lowering complex part to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Substring &substring) {
    TODO(getLoc(), "lowering substrings to HLFIR");
  }

  mlir::Type visit(const Fortran::evaluate::SymbolRef &symbolRef,
                   PartInfo &partInfo) {
    partInfo.base = gen(symbolRef);
    hlfir::genLengthParameters(getLoc(), getBuilder(), partInfo.base,
                               partInfo.typeParams);
    return partInfo.base.getElementOrSequenceType();
  }

  mlir::Type visit(const Fortran::evaluate::ArrayRef &arrayRef,
                   PartInfo &partInfo) {
    mlir::Type baseType;
    if (const auto *component = arrayRef.base().UnwrapComponent())
      baseType = visit(*component, partInfo);
    baseType = visit(arrayRef.base().GetLastSymbol(), partInfo);

    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    mlir::Type idxTy = builder.getIndexType();
    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> bounds;
    auto getBounds = [&](unsigned i) {
      if (bounds.empty())
        bounds = hlfir::genBounds(loc, builder, partInfo.base);
      return bounds[i];
    };
    auto frontEndResultShape =
        Fortran::evaluate::GetShape(converter.getFoldingContext(), arrayRef);
    llvm::SmallVector<mlir::Value> resultExtents;
    fir::SequenceType::Shape resultTypeShape;
    for (auto subscript : llvm::enumerate(arrayRef.subscript())) {
      if (const auto *triplet =
              std::get_if<Fortran::evaluate::Triplet>(&subscript.value().u)) {
        mlir::Value lb, ub;
        if (const auto &lbExpr = triplet->lower())
          lb = genSubscript(*lbExpr);
        else
          lb = getBounds(subscript.index()).first;
        if (const auto &ubExpr = triplet->upper())
          ub = genSubscript(*ubExpr);
        else
          ub = getBounds(subscript.index()).second;
        lb = builder.createConvert(loc, idxTy, lb);
        ub = builder.createConvert(loc, idxTy, ub);
        mlir::Value stride = genSubscript(triplet->stride());
        stride = builder.createConvert(loc, idxTy, stride);
        mlir::Value extent;
        // Use constant extent if possible. The main advantage to do this now
        // is to get the best FIR array types as possible while lowering.
        if (frontEndResultShape)
          if (auto maybeI64 = Fortran::evaluate::ToInt64(
                  frontEndResultShape->at(resultExtents.size()))) {
            resultTypeShape.push_back(*maybeI64);
            extent = builder.createIntegerConstant(loc, idxTy, *maybeI64);
          }
        if (!extent) {
          extent = builder.genExtentFromTriplet(loc, lb, ub, stride, idxTy);
          resultTypeShape.push_back(fir::SequenceType::getUnknownExtent());
        }
        partInfo.subscripts.emplace_back(
            hlfir::DesignateOp::Triplet{lb, ub, stride});
        resultExtents.push_back(extent);
      } else {
        const auto &expr =
            std::get<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
                subscript.value().u)
                .value();
        if (expr.Rank() > 0)
          TODO(getLoc(), "vector subscripts in HLFIR");
        partInfo.subscripts.push_back(genSubscript(expr));
      }
    }

    assert(resultExtents.size() == resultTypeShape.size() &&
           "inconsistent hlfir.designate shape");
    mlir::Type resultType = baseType.cast<fir::SequenceType>().getEleTy();
    if (!resultTypeShape.empty()) {
      resultType = fir::SequenceType::get(resultTypeShape, resultType);
      partInfo.resultShape = builder.genShape(loc, resultExtents);
    }
    return resultType;
  }

  mlir::Type visit(const Fortran::evaluate::Component &component,
                   PartInfo &partInfo) {
    TODO(getLoc(), "lowering component to HLFIR");
  }

  /// Lower a subscript expression. If it is a scalar subscript that is
  /// a variable, it is loaded into an integer value.
  template <typename T>
  hlfir::EntityWithAttributes
  genSubscript(const Fortran::evaluate::Expr<T> &expr);

  mlir::Location getLoc() const { return loc; }
  Fortran::lower::AbstractConverter &getConverter() { return converter; }
  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }
  Fortran::lower::SymMap &getSymMap() { return symMap; }
  Fortran::lower::StatementContext &getStmtCtx() { return stmtCtx; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
};

//===--------------------------------------------------------------------===//
// Binary Operation implementation
//===--------------------------------------------------------------------===//

template <typename T>
struct BinaryOp {};

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  struct BinaryOp<Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type<       \
      Fortran::common::TypeCategory::GenBinTyCat, KIND>>> {                    \
    using Op = Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type<          \
        Fortran::common::TypeCategory::GenBinTyCat, KIND>>;                    \
    static hlfir::EntityWithAttributes gen(mlir::Location loc,                 \
                                           fir::FirOpBuilder &builder,         \
                                           const Op &, hlfir::Entity lhs,      \
                                           hlfir::Entity rhs) {                \
      return hlfir::EntityWithAttributes{                                      \
          builder.create<GenBinFirOp>(loc, lhs, rhs)};                         \
    }                                                                          \
  };

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
struct BinaryOp<Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>>> {
  using Op = Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs, hlfir::Entity rhs) {
    mlir::Type ty = Fortran::lower::getFIRType(builder.getContext(), TC, KIND,
                                               /*params=*/std::nullopt);
    return hlfir::EntityWithAttributes{
        Fortran::lower::genPow(builder, loc, ty, lhs, rhs)};
  }
};

template <Fortran::common::TypeCategory TC, int KIND>
struct BinaryOp<
    Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>> {
  using Op =
      Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs, hlfir::Entity rhs) {
    mlir::Type ty = Fortran::lower::getFIRType(builder.getContext(), TC, KIND,
                                               /*params=*/std::nullopt);
    return hlfir::EntityWithAttributes{
        Fortran::lower::genPow(builder, loc, ty, lhs, rhs)};
  }
};

template <Fortran::common::TypeCategory TC, int KIND>
struct BinaryOp<
    Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>> {
  using Op = Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs,
                                         hlfir::Entity rhs) {
    // evaluate::Extremum is only created by the front-end when building
    // compiler generated expressions (like when folding LEN() or shape/bounds
    // inquiries). MIN and MAX are represented as evaluate::ProcedureRef and are
    // not going through here. So far the frontend does not generate character
    // Extremum so there is no way to test it.
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      fir::emitFatalError(loc, "Fortran::evaluate::Extremum are unexpected");
    }
    llvm::SmallVector<mlir::Value, 2> args{lhs, rhs};
    fir::ExtendedValue res = op.ordering == Fortran::evaluate::Ordering::Greater
                                 ? Fortran::lower::genMax(builder, loc, args)
                                 : Fortran::lower::genMin(builder, loc, args);
    return hlfir::EntityWithAttributes{fir::getBase(res)};
  }
};

/// Convert parser's INTEGER relational operators to MLIR.
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

template <int KIND>
struct BinaryOp<Fortran::evaluate::Relational<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, KIND>>> {
  using Op = Fortran::evaluate::Relational<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs,
                                         hlfir::Entity rhs) {
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, translateRelational(op.opr), lhs, rhs);
    return hlfir::EntityWithAttributes{cmp};
  }
};

template <int KIND>
struct BinaryOp<Fortran::evaluate::Relational<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>>> {
  using Op = Fortran::evaluate::Relational<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs,
                                         hlfir::Entity rhs) {
    auto cmp = builder.create<mlir::arith::CmpFOp>(
        loc, translateFloatRelational(op.opr), lhs, rhs);
    return hlfir::EntityWithAttributes{cmp};
  }
};

template <int KIND>
struct BinaryOp<Fortran::evaluate::Relational<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Complex, KIND>>> {
  using Op = Fortran::evaluate::Relational<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Complex, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs,
                                         hlfir::Entity rhs) {
    auto cmp = builder.create<fir::CmpcOp>(
        loc, translateFloatRelational(op.opr), lhs, rhs);
    return hlfir::EntityWithAttributes{cmp};
  }
};

template <int KIND>
struct BinaryOp<Fortran::evaluate::Relational<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>> {
  using Op = Fortran::evaluate::Relational<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs,
                                         hlfir::Entity rhs) {
    auto [lhsExv, lhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, lhs);
    auto [rhsExv, rhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, rhs);
    auto cmp = fir::runtime::genCharCompare(
        builder, loc, translateRelational(op.opr), lhsExv, rhsExv);
    if (lhsCleanUp)
      lhsCleanUp.value()();
    if (rhsCleanUp)
      rhsCleanUp.value()();
    return hlfir::EntityWithAttributes{cmp};
  }
};

template <int KIND>
struct BinaryOp<Fortran::evaluate::LogicalOperation<KIND>> {
  using Op = Fortran::evaluate::LogicalOperation<KIND>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs,
                                         hlfir::Entity rhs) {
    mlir::Type i1Type = builder.getI1Type();
    mlir::Value i1Lhs = builder.createConvert(loc, i1Type, lhs);
    mlir::Value i1Rhs = builder.createConvert(loc, i1Type, rhs);
    switch (op.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return hlfir::EntityWithAttributes{
          builder.create<mlir::arith::AndIOp>(loc, i1Lhs, i1Rhs)};
    case Fortran::evaluate::LogicalOperator::Or:
      return hlfir::EntityWithAttributes{
          builder.create<mlir::arith::OrIOp>(loc, i1Lhs, i1Rhs)};
    case Fortran::evaluate::LogicalOperator::Eqv:
      return hlfir::EntityWithAttributes{builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, i1Lhs, i1Rhs)};
    case Fortran::evaluate::LogicalOperator::Neqv:
      return hlfir::EntityWithAttributes{builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, i1Lhs, i1Rhs)};
    case Fortran::evaluate::LogicalOperator::Not:
      // lib/evaluate expression for .NOT. is Fortran::evaluate::Not<KIND>.
      llvm_unreachable(".NOT. is not a binary operator");
    }
    llvm_unreachable("unhandled logical operation");
  }
};

template <int KIND>
struct BinaryOp<Fortran::evaluate::ComplexConstructor<KIND>> {
  using Op = Fortran::evaluate::ComplexConstructor<KIND>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs, hlfir::Entity rhs) {
    mlir::Value res =
        fir::factory::Complex{builder, loc}.createComplex(KIND, lhs, rhs);
    return hlfir::EntityWithAttributes{res};
  }
};

template <int KIND>
struct BinaryOp<Fortran::evaluate::SetLength<KIND>> {
  using Op = Fortran::evaluate::SetLength<KIND>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &, const Op &,
                                         hlfir::Entity, hlfir::Entity) {
    TODO(loc, "SetLength lowering to HLFIR");
  }
};

//===--------------------------------------------------------------------===//
// Unary Operation implementation
//===--------------------------------------------------------------------===//

template <typename T>
struct UnaryOp {};

template <int KIND>
struct UnaryOp<Fortran::evaluate::Not<KIND>> {
  using Op = Fortran::evaluate::Not<KIND>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs) {
    mlir::Value one = builder.createBool(loc, true);
    mlir::Value val = builder.createConvert(loc, builder.getI1Type(), lhs);
    return hlfir::EntityWithAttributes{
        builder.create<mlir::arith::XOrIOp>(loc, val, one)};
  }
};

template <int KIND>
struct UnaryOp<Fortran::evaluate::Negate<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, KIND>>> {
  using Op = Fortran::evaluate::Negate<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs) {
    // Like LLVM, integer negation is the binary op "0 - value"
    mlir::Type type = Fortran::lower::getFIRType(
        builder.getContext(), Fortran::common::TypeCategory::Integer, KIND,
        /*params=*/std::nullopt);
    mlir::Value zero = builder.createIntegerConstant(loc, type, 0);
    return hlfir::EntityWithAttributes{
        builder.create<mlir::arith::SubIOp>(loc, zero, lhs)};
  }
};

template <int KIND>
struct UnaryOp<Fortran::evaluate::Negate<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>>> {
  using Op = Fortran::evaluate::Negate<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs) {
    return hlfir::EntityWithAttributes{
        builder.create<mlir::arith::NegFOp>(loc, lhs)};
  }
};

template <int KIND>
struct UnaryOp<Fortran::evaluate::Negate<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Complex, KIND>>> {
  using Op = Fortran::evaluate::Negate<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Complex, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs) {
    return hlfir::EntityWithAttributes{builder.create<fir::NegcOp>(loc, lhs)};
  }
};

template <int KIND>
struct UnaryOp<Fortran::evaluate::ComplexComponent<KIND>> {
  using Op = Fortran::evaluate::ComplexComponent<KIND>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs) {
    mlir::Value res = fir::factory::Complex{builder, loc}.extractComplexPart(
        lhs, op.isImaginaryPart);
    return hlfir::EntityWithAttributes{res};
  }
};

template <typename T>
struct UnaryOp<Fortran::evaluate::Parentheses<T>> {
  using Op = Fortran::evaluate::Parentheses<T>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         const Op &op, hlfir::Entity lhs) {
    if (lhs.isVariable())
      return hlfir::EntityWithAttributes{
          builder.create<hlfir::AsExprOp>(loc, lhs)};
    return hlfir::EntityWithAttributes{
        builder.create<hlfir::NoReassocOp>(loc, lhs.getType(), lhs)};
  }
};

template <Fortran::common::TypeCategory TC1, int KIND,
          Fortran::common::TypeCategory TC2>
struct UnaryOp<
    Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>, TC2>> {
  using Op =
      Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>, TC2>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs) {
    if constexpr (TC1 == Fortran::common::TypeCategory::Character &&
                  TC2 == TC1) {
      TODO(loc, "character conversion in HLFIR");
    }
    mlir::Type type = Fortran::lower::getFIRType(builder.getContext(), TC1,
                                                 KIND, /*params=*/std::nullopt);
    mlir::Value res = builder.convertWithSemantics(loc, type, lhs);
    return hlfir::EntityWithAttributes{res};
  }
};

/// Lower Expr to HLFIR.
class HlfirBuilder {
public:
  HlfirBuilder(mlir::Location loc, Fortran::lower::AbstractConverter &converter,
               Fortran::lower::SymMap &symMap,
               Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, symMap{symMap}, stmtCtx{stmtCtx}, loc{loc} {}

  template <typename T>
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::Expr<T> &expr) {
    return std::visit([&](const auto &x) { return gen(x); }, expr.u);
  }

private:
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::BOZLiteralConstant &expr) {
    fir::emitFatalError(loc, "BOZ literal must be replaced by semantics");
  }
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::NullPointer &expr) {
    TODO(getLoc(), "lowering NullPointer to HLFIR");
  }
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ProcedureDesignator &expr) {
    TODO(getLoc(), "lowering ProcDes to HLFIR");
  }
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::ProcedureRef &expr) {
    TODO(getLoc(), "lowering ProcRef to HLFIR");
  }

  template <typename T>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Designator<T> &designator) {
    return HlfirDesignatorBuilder(getLoc(), getConverter(), getSymMap(),
                                  getStmtCtx())
        .gen(designator.u);
  }

  template <typename T>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::FunctionRef<T> &expr) {
    mlir::Type resType =
        Fortran::lower::TypeBuilder<T>::genType(getConverter(), expr);
    return Fortran::lower::convertCallToHLFIR(getLoc(), getConverter(), expr,
                                              resType, getSymMap(),
                                              getStmtCtx())
        .value();
  }

  template <typename T>
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::Constant<T> &expr) {
    mlir::Location loc = getLoc();
    if constexpr (std::is_same_v<T, Fortran::evaluate::SomeDerived>) {
      TODO(loc, "lowering derived type constant to HLFIR");
    } else {
      fir::FirOpBuilder &builder = getBuilder();
      fir::ExtendedValue exv =
          Fortran::lower::IntrinsicConstantBuilder<T::category, T::kind>::gen(
              builder, loc, expr, /*outlineBigConstantInReadOnlyMemory=*/true);
      if (const auto *scalarBox = exv.getUnboxed())
        if (fir::isa_trivial(scalarBox->getType()))
          return hlfir::EntityWithAttributes(*scalarBox);
      if (auto addressOf = fir::getBase(exv).getDefiningOp<fir::AddrOfOp>()) {
        auto flags = fir::FortranVariableFlagsAttr::get(
            builder.getContext(), fir::FortranVariableFlagsEnum::parameter);
        return hlfir::genDeclare(
            loc, builder, exv,
            addressOf.getSymbol().getRootReference().getValue(), flags);
      }
      fir::emitFatalError(loc, "Constant<T> was lowered to unexpected format");
    }
  }

  template <typename T>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ArrayConstructor<T> &expr) {
    TODO(getLoc(), "lowering ArrayCtor to HLFIR");
  }

  template <typename D, typename R, typename O>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Operation<D, R, O> &op) {
    auto &builder = getBuilder();
    mlir::Location loc = getLoc();
    if (op.Rank() != 0)
      TODO(loc, "elemental operations in HLFIR");
    auto left = hlfir::loadTrivialScalar(loc, builder, gen(op.left()));
    return UnaryOp<D>::gen(loc, builder, op.derived(), left);
  }

  template <typename D, typename R, typename LO, typename RO>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    auto &builder = getBuilder();
    mlir::Location loc = getLoc();
    if (op.Rank() != 0)
      TODO(loc, "elemental operations in HLFIR");
    auto left = hlfir::loadTrivialScalar(loc, builder, gen(op.left()));
    auto right = hlfir::loadTrivialScalar(loc, builder, gen(op.right()));
    return BinaryOp<D>::gen(loc, builder, op.derived(), left, right);
  }

  template <int KIND>
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::Concat<KIND> &op) {
    auto lhs = gen(op.left());
    auto rhs = gen(op.right());
    llvm::SmallVector<mlir::Value> lengths;
    auto &builder = getBuilder();
    mlir::Location loc = getLoc();
    hlfir::genLengthParameters(loc, builder, lhs, lengths);
    hlfir::genLengthParameters(loc, builder, rhs, lengths);
    assert(lengths.size() == 2 && "lacks rhs or lhs length");
    mlir::Type idxType = builder.getIndexType();
    mlir::Value lhsLen = builder.createConvert(loc, idxType, lengths[0]);
    mlir::Value rhsLen = builder.createConvert(loc, idxType, lengths[1]);
    mlir::Value len = builder.create<mlir::arith::AddIOp>(loc, lhsLen, rhsLen);
    auto concat =
        builder.create<hlfir::ConcatOp>(loc, mlir::ValueRange{lhs, rhs}, len);
    return hlfir::EntityWithAttributes{concat.getResult()};
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return gen(x); }, op.u);
  }

  hlfir::EntityWithAttributes gen(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "lowering type parameter inquiry to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::DescriptorInquiry &desc) {
    TODO(getLoc(), "lowering descriptor inquiry to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ImpliedDoIndex &var) {
    TODO(getLoc(), "lowering implied do index to HLFIR");
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::StructureConstructor &var) {
    TODO(getLoc(), "lowering structure constructor to HLFIR");
  }

  mlir::Location getLoc() const { return loc; }
  Fortran::lower::AbstractConverter &getConverter() { return converter; }
  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }
  Fortran::lower::SymMap &getSymMap() { return symMap; }
  Fortran::lower::StatementContext &getStmtCtx() { return stmtCtx; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
};

template <typename T>
hlfir::EntityWithAttributes
HlfirDesignatorBuilder::genSubscript(const Fortran::evaluate::Expr<T> &expr) {
  auto loweredExpr =
      HlfirBuilder(getLoc(), getConverter(), getSymMap(), getStmtCtx())
          .gen(expr);
  if (!loweredExpr.isArray()) {
    fir::FirOpBuilder &builder = getBuilder();
    if (loweredExpr.isVariable())
      return hlfir::EntityWithAttributes{
          hlfir::loadTrivialScalar(loc, builder, loweredExpr).getBase()};
    // Skip constant conversions that litters designators and makes generated
    // IR harder to read: directly use index constants for constant subscripts.
    mlir::Type idxTy = builder.getIndexType();
    if (loweredExpr.getType() != idxTy)
      if (auto cstIndex = fir::factory::getIntIfConstant(loweredExpr))
        return hlfir::EntityWithAttributes{
            builder.createIntegerConstant(getLoc(), idxTy, *cstIndex)};
  }
  return loweredExpr;
}

} // namespace

hlfir::EntityWithAttributes Fortran::lower::convertExprToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  return HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
}
