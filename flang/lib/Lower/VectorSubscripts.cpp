//===-- VectorSubscripts.cpp -- Vector subscripts tools -------------------===//
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

#include "flang/Lower/VectorSubscripts.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Semantics/expression.h"

namespace {
/// Helper class to lower a designator containing vector subscripts into a
/// lowered representation that can be worked with.
class VectorSubscriptBoxBuilder {
public:
  VectorSubscriptBoxBuilder(mlir::Location loc,
                            Fortran::lower::AbstractConverter &converter,
                            Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, stmtCtx{stmtCtx}, loc{loc} {}

  Fortran::lower::VectorSubscriptBox gen(const Fortran::lower::SomeExpr &expr) {
    elementType = genDesignator(expr);
    return Fortran::lower::VectorSubscriptBox(
        std::move(loweredBase), std::move(loweredSubscripts),
        std::move(componentPath), substringBounds, elementType);
  }

private:
  using LoweredVectorSubscript =
      Fortran::lower::VectorSubscriptBox::LoweredVectorSubscript;
  using LoweredTriplet = Fortran::lower::VectorSubscriptBox::LoweredTriplet;
  using LoweredSubscript = Fortran::lower::VectorSubscriptBox::LoweredSubscript;
  using MaybeSubstring = Fortran::lower::VectorSubscriptBox::MaybeSubstring;

  /// genDesignator unwraps a Designator<T> and calls `gen` on what the
  /// designator actually contains.
  template <typename A>
  mlir::Type genDesignator(const A &) {
    fir::emitFatalError(loc, "expr must contain a designator");
  }
  template <typename T>
  mlir::Type genDesignator(const Fortran::evaluate::Expr<T> &expr) {
    using ExprVariant = decltype(Fortran::evaluate::Expr<T>::u);
    using Designator = Fortran::evaluate::Designator<T>;
    if constexpr (Fortran::common::HasMember<Designator, ExprVariant>) {
      const auto &designator = std::get<Designator>(expr.u);
      return std::visit([&](const auto &x) { return gen(x); }, designator.u);
    } else {
      return std::visit([&](const auto &x) { return genDesignator(x); },
                        expr.u);
    }
  }

  // The gen(X) methods visit X to lower its base and subscripts and return the
  // type of X elements.

  mlir::Type gen(const Fortran::evaluate::DataRef &dataRef) {
    return std::visit([&](const auto &ref) -> mlir::Type { return gen(ref); },
                      dataRef.u);
  }

  mlir::Type gen(const Fortran::evaluate::SymbolRef &symRef) {
    // Never visited because expr lowering is used to lowered the ranked
    // ArrayRef.
    fir::emitFatalError(
        loc, "expected at least one ArrayRef with vector susbcripts");
  }

  mlir::Type gen(const Fortran::evaluate::Substring &substring) {
    // StaticDataObject::Pointer bases are constants and cannot be
    // subscripted, so the base must be a DataRef here.
    mlir::Type baseElementType =
        gen(std::get<Fortran::evaluate::DataRef>(substring.parent()));
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Type idxTy = builder.getIndexType();
    mlir::Value lb = genScalarValue(substring.lower());
    substringBounds.emplace_back(builder.createConvert(loc, idxTy, lb));
    if (const auto &ubExpr = substring.upper()) {
      mlir::Value ub = genScalarValue(*ubExpr);
      substringBounds.emplace_back(builder.createConvert(loc, idxTy, ub));
    }
    return baseElementType;
  }

  mlir::Type gen(const Fortran::evaluate::ComplexPart &complexPart) {
    auto complexType = gen(complexPart.complex());
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Type i32Ty = builder.getI32Type(); // llvm's GEP requires i32
    mlir::Value offset = builder.createIntegerConstant(
        loc, i32Ty,
        complexPart.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    componentPath.emplace_back(offset);
    return fir::factory::Complex{builder, loc}.getComplexPartType(complexType);
  }

  mlir::Type gen(const Fortran::evaluate::Component &component) {
    auto recTy = gen(component.base()).cast<fir::RecordType>();
    const Fortran::semantics::Symbol &componentSymbol =
        component.GetLastSymbol();
    // Parent components will not be found here, they are not part
    // of the FIR type and cannot be used in the path yet.
    if (componentSymbol.test(Fortran::semantics::Symbol::Flag::ParentComp))
      TODO(loc, "reference to parent component");
    mlir::Type fldTy = fir::FieldType::get(&converter.getMLIRContext());
    llvm::StringRef componentName = toStringRef(componentSymbol.name());
    // Parameters threading in field_index is not yet very clear. We only
    // have the ones of the ranked array ref at hand, but it looks like
    // the fir.field_index expects the one of the direct base.
    if (recTy.getNumLenParams() != 0)
      TODO(loc, "threading length parameters in field index op");
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    componentPath.emplace_back(builder.create<fir::FieldIndexOp>(
        loc, fldTy, componentName, recTy, /*typeParams*/ llvm::None));
    return fir::unwrapSequenceType(recTy.getType(componentName));
  }

  mlir::Type gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    auto isTripletOrVector =
        [](const Fortran::evaluate::Subscript &subscript) -> bool {
      return std::visit(
          Fortran::common::visitors{
              [](const Fortran::evaluate::IndirectSubscriptIntegerExpr &expr) {
                return expr.value().Rank() != 0;
              },
              [&](const Fortran::evaluate::Triplet &) { return true; }},
          subscript.u);
    };
    if (llvm::any_of(arrayRef.subscript(), isTripletOrVector))
      return genRankedArrayRefSubscriptAndBase(arrayRef);

    // This is a scalar ArrayRef (only scalar indexes), collect the indexes and
    // visit the base that must contain another arrayRef with the vector
    // subscript.
    mlir::Type elementType = gen(namedEntityToDataRef(arrayRef.base()));
    for (const Fortran::evaluate::Subscript &subscript : arrayRef.subscript()) {
      const auto &expr =
          std::get<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
              subscript.u);
      componentPath.emplace_back(genScalarValue(expr.value()));
    }
    return elementType;
  }

  /// Lower the subscripts and base of the ArrayRef that is an array (there must
  /// be one since there is a vector subscript, and there can only be one
  /// according to C925).
  mlir::Type genRankedArrayRefSubscriptAndBase(
      const Fortran::evaluate::ArrayRef &arrayRef) {
    // Lower the save the base
    Fortran::lower::SomeExpr baseExpr = namedEntityToExpr(arrayRef.base());
    loweredBase = converter.genExprAddr(baseExpr, stmtCtx);
    // Lower and save the subscripts
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Type idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    for (const auto &subscript : llvm::enumerate(arrayRef.subscript())) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &expr) {
                if (expr.value().Rank() == 0) {
                  // Simple scalar subscript
                  loweredSubscripts.emplace_back(genScalarValue(expr.value()));
                } else {
                  // Vector subscript.
                  // Remove conversion if any to avoid temp creation that may
                  // have been added by the front-end to avoid the creation of a
                  // temp array value.
                  auto vector = converter.genExprAddr(
                      ignoreEvConvert(expr.value()), stmtCtx);
                  mlir::Value size =
                      fir::factory::readExtent(builder, loc, vector, /*dim=*/0);
                  size = builder.createConvert(loc, idxTy, size);
                  loweredSubscripts.emplace_back(
                      LoweredVectorSubscript{std::move(vector), size});
                }
              },
              [&](const Fortran::evaluate::Triplet &triplet) {
                mlir::Value lb, ub;
                if (const auto &lbExpr = triplet.lower())
                  lb = genScalarValue(*lbExpr);
                else
                  lb = fir::factory::readLowerBound(builder, loc, loweredBase,
                                                    subscript.index(), one);
                if (const auto &ubExpr = triplet.upper())
                  ub = genScalarValue(*ubExpr);
                else
                  ub = fir::factory::readExtent(builder, loc, loweredBase,
                                                subscript.index());
                lb = builder.createConvert(loc, idxTy, lb);
                ub = builder.createConvert(loc, idxTy, ub);
                mlir::Value stride = genScalarValue(triplet.stride());
                stride = builder.createConvert(loc, idxTy, stride);
                loweredSubscripts.emplace_back(LoweredTriplet{lb, ub, stride});
              },
          },
          subscript.value().u);
    }
    return fir::unwrapSequenceType(
        fir::unwrapPassByRefType(fir::getBase(loweredBase).getType()));
  }

  mlir::Type gen(const Fortran::evaluate::CoarrayRef &) {
    // Is this possible/legal ?
    TODO(loc, "coarray ref with vector subscript in IO input");
  }

  template <typename A>
  mlir::Value genScalarValue(const A &expr) {
    return fir::getBase(converter.genExprValue(toEvExpr(expr), stmtCtx));
  }

  Fortran::evaluate::DataRef
  namedEntityToDataRef(const Fortran::evaluate::NamedEntity &namedEntity) {
    if (namedEntity.IsSymbol())
      return Fortran::evaluate::DataRef{namedEntity.GetFirstSymbol()};
    return Fortran::evaluate::DataRef{namedEntity.GetComponent()};
  }

  Fortran::lower::SomeExpr
  namedEntityToExpr(const Fortran::evaluate::NamedEntity &namedEntity) {
    return Fortran::evaluate::AsGenericExpr(namedEntityToDataRef(namedEntity))
        .value();
  }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::StatementContext &stmtCtx;
  mlir::Location loc;
  /// Elements of VectorSubscriptBox being built.
  fir::ExtendedValue loweredBase;
  llvm::SmallVector<LoweredSubscript, 16> loweredSubscripts;
  llvm::SmallVector<mlir::Value> componentPath;
  MaybeSubstring substringBounds;
  mlir::Type elementType;
};
} // namespace

Fortran::lower::VectorSubscriptBox Fortran::lower::genVectorSubscriptBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::lower::SomeExpr &expr) {
  return VectorSubscriptBoxBuilder(loc, converter, stmtCtx).gen(expr);
}

template <typename LoopType, typename Generator>
mlir::Value Fortran::lower::VectorSubscriptBox::loopOverElementsBase(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Generator &elementalGenerator,
    [[maybe_unused]] mlir::Value initialCondition) {
  mlir::Value shape = builder.createShape(loc, loweredBase);
  mlir::Value slice = createSlice(builder, loc);

  // Create loop nest for triplets and vector subscripts in column
  // major order.
  llvm::SmallVector<mlir::Value> inductionVariables;
  LoopType outerLoop;
  for (auto [lb, ub, step] : genLoopBounds(builder, loc)) {
    LoopType loop;
    if constexpr (std::is_same_v<LoopType, fir::IterWhileOp>) {
      loop =
          builder.create<fir::IterWhileOp>(loc, lb, ub, step, initialCondition);
      initialCondition = loop.getIterateVar();
      if (!outerLoop)
        outerLoop = loop;
      else
        builder.create<fir::ResultOp>(loc, loop.getResult(0));
    } else {
      loop =
          builder.create<fir::DoLoopOp>(loc, lb, ub, step, /*unordered=*/false);
      if (!outerLoop)
        outerLoop = loop;
    }
    builder.setInsertionPointToStart(loop.getBody());
    inductionVariables.push_back(loop.getInductionVar());
  }
  assert(outerLoop && !inductionVariables.empty() &&
         "at least one loop should be created");

  fir::ExtendedValue elem =
      getElementAt(builder, loc, shape, slice, inductionVariables);

  if constexpr (std::is_same_v<LoopType, fir::IterWhileOp>) {
    auto res = elementalGenerator(elem);
    builder.create<fir::ResultOp>(loc, res);
    builder.setInsertionPointAfter(outerLoop);
    return outerLoop.getResult(0);
  } else {
    elementalGenerator(elem);
    builder.setInsertionPointAfter(outerLoop);
    return {};
  }
}

void Fortran::lower::VectorSubscriptBox::loopOverElements(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const ElementalGenerator &elementalGenerator) {
  mlir::Value initialCondition;
  loopOverElementsBase<fir::DoLoopOp, ElementalGenerator>(
      builder, loc, elementalGenerator, initialCondition);
}

mlir::Value Fortran::lower::VectorSubscriptBox::loopOverElementsWhile(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const ElementalGeneratorWithBoolReturn &elementalGenerator,
    mlir::Value initialCondition) {
  return loopOverElementsBase<fir::IterWhileOp,
                              ElementalGeneratorWithBoolReturn>(
      builder, loc, elementalGenerator, initialCondition);
}

mlir::Value
Fortran::lower::VectorSubscriptBox::createSlice(fir::FirOpBuilder &builder,
                                                mlir::Location loc) {
  mlir::Type idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> triples;
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  auto undef = builder.create<fir::UndefOp>(loc, idxTy);
  for (const LoweredSubscript &subscript : loweredSubscripts)
    std::visit(Fortran::common::visitors{
                   [&](const LoweredTriplet &triplet) {
                     triples.emplace_back(triplet.lb);
                     triples.emplace_back(triplet.ub);
                     triples.emplace_back(triplet.stride);
                   },
                   [&](const LoweredVectorSubscript &vector) {
                     triples.emplace_back(one);
                     triples.emplace_back(vector.size);
                     triples.emplace_back(one);
                   },
                   [&](const mlir::Value &i) {
                     triples.emplace_back(i);
                     triples.emplace_back(undef);
                     triples.emplace_back(undef);
                   },
               },
               subscript);
  return builder.create<fir::SliceOp>(loc, triples, componentPath);
}

llvm::SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Value>>
Fortran::lower::VectorSubscriptBox::genLoopBounds(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) {
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  llvm::SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Value>> bounds;
  size_t dimension = loweredSubscripts.size();
  for (const LoweredSubscript &subscript : llvm::reverse(loweredSubscripts)) {
    --dimension;
    if (std::holds_alternative<mlir::Value>(subscript))
      continue;
    mlir::Value lb, ub, step;
    if (const auto *triplet = std::get_if<LoweredTriplet>(&subscript)) {
      mlir::Value extent = builder.genExtentFromTriplet(
          loc, triplet->lb, triplet->ub, triplet->stride, idxTy);
      mlir::Value baseLb = fir::factory::readLowerBound(
          builder, loc, loweredBase, dimension, one);
      baseLb = builder.createConvert(loc, idxTy, baseLb);
      lb = baseLb;
      ub = builder.create<mlir::arith::SubIOp>(loc, idxTy, extent, one);
      ub = builder.create<mlir::arith::AddIOp>(loc, idxTy, ub, baseLb);
      step = one;
    } else {
      const auto &vector = std::get<LoweredVectorSubscript>(subscript);
      lb = zero;
      ub = builder.create<mlir::arith::SubIOp>(loc, idxTy, vector.size, one);
      step = one;
    }
    bounds.emplace_back(lb, ub, step);
  }
  return bounds;
}

fir::ExtendedValue Fortran::lower::VectorSubscriptBox::getElementAt(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value shape,
    mlir::Value slice, mlir::ValueRange inductionVariables) {
  /// Generate the indexes for the array_coor inside the loops.
  mlir::Type idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> indexes;
  size_t inductionIdx = inductionVariables.size() - 1;
  for (const LoweredSubscript &subscript : loweredSubscripts)
    std::visit(Fortran::common::visitors{
                   [&](const LoweredTriplet &triplet) {
                     indexes.emplace_back(inductionVariables[inductionIdx--]);
                   },
                   [&](const LoweredVectorSubscript &vector) {
                     mlir::Value vecIndex = inductionVariables[inductionIdx--];
                     mlir::Value vecBase = fir::getBase(vector.vector);
                     mlir::Type vecEleTy = fir::unwrapSequenceType(
                         fir::unwrapPassByRefType(vecBase.getType()));
                     mlir::Type refTy = builder.getRefType(vecEleTy);
                     auto vecEltRef = builder.create<fir::CoordinateOp>(
                         loc, refTy, vecBase, vecIndex);
                     auto vecElt =
                         builder.create<fir::LoadOp>(loc, vecEleTy, vecEltRef);
                     indexes.emplace_back(
                         builder.createConvert(loc, idxTy, vecElt));
                   },
                   [&](const mlir::Value &i) {
                     indexes.emplace_back(builder.createConvert(loc, idxTy, i));
                   },
               },
               subscript);
  mlir::Type refTy = builder.getRefType(getElementType());
  auto elementAddr = builder.create<fir::ArrayCoorOp>(
      loc, refTy, fir::getBase(loweredBase), shape, slice, indexes,
      fir::getTypeParams(loweredBase));
  fir::ExtendedValue element = fir::factory::arraySectionElementToExtendedValue(
      builder, loc, loweredBase, elementAddr, slice);
  if (!substringBounds.empty()) {
    const fir::CharBoxValue *charBox = element.getCharBox();
    assert(charBox && "substring requires CharBox base");
    fir::factory::CharacterExprHelper helper{builder, loc};
    return helper.createSubstring(*charBox, substringBounds);
  }
  return element;
}
