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
#include "flang/Lower/ConvertConstant.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
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
    TODO(getLoc(), "lowering funcRef to HLFIR");
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

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>, TC2>
          &convert) {
    TODO(getLoc(), "lowering convert to HLFIR");
  }

  template <typename D, typename R, typename O>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Operation<D, R, O> &op) {
    TODO(getLoc(), "lowering unary op to HLFIR");
  }

  template <typename D, typename R, typename LO, typename RO>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    TODO(getLoc(), "lowering binary op to HLFIR");
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
