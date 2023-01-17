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
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

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

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::NamedEntity &namedEntity) {
    if (namedEntity.IsSymbol())
      return gen(Fortran::evaluate::SymbolRef{namedEntity.GetLastSymbol()});
    return gen(namedEntity.GetComponent());
  }

private:
  /// Struct that is filled while visiting a part-ref (in the "visit" member
  /// function) before the top level "gen" generates an hlfir.declare for the
  /// part ref. It contains the lowered pieces of the part-ref that will
  /// become the operands of an hlfir.declare.
  struct PartInfo {
    fir::FortranVariableOpInterface base;
    std::string componentName{};
    mlir::Value componentShape;
    hlfir::DesignateOp::Subscripts subscripts;
    mlir::Value resultShape;
    llvm::SmallVector<mlir::Value> typeParams;
    llvm::SmallVector<mlir::Value, 2> substring;
  };

  // Given the value type of a designator (T or fir.array<T>) and the front-end
  // node for the designator, compute the memory type (fir.class, fir.ref, or
  // fir.box)...
  template <typename T>
  mlir::Type computeDesignatorType(mlir::Type resultValueType,
                                   const PartInfo &partInfo,
                                   const T &designatorNode) {
    // Dynamic type of polymorphic base must be kept if the designator is
    // polymorphic.
    if (isPolymorphic(designatorNode))
      return fir::ClassType::get(resultValueType);
    // Character scalar with dynamic length needs a fir.boxchar to hold the
    // designator length.
    auto charType = resultValueType.dyn_cast<fir::CharacterType>();
    if (charType && charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());
    // Arrays with non default lower bounds or dynamic length or dynamic extent
    // need a fir.box to hold the dynamic or lower bound information.
    if (fir::hasDynamicSize(resultValueType) ||
        hasNonDefaultLowerBounds(partInfo))
      return fir::BoxType::get(resultValueType);
    // Non simply contiguous ref require a fir.box to carry the byte stride.
    if (resultValueType.isa<fir::SequenceType>() &&
        !Fortran::evaluate::IsSimplyContiguous(
            designatorNode, getConverter().getFoldingContext()))
      return fir::BoxType::get(resultValueType);
    // Other designators can be handled as raw addresses.
    return fir::ReferenceType::get(resultValueType);
  }

  template <typename T>
  static bool isPolymorphic(const T &designatorNode) {
    if constexpr (!std::is_same_v<T, Fortran::evaluate::Substring>) {
      return Fortran::semantics::IsPolymorphic(designatorNode.GetLastSymbol());
    }
    return false;
  }

  template <typename T>
  /// Generate an hlfir.designate for a part-ref given a filled PartInfo and the
  /// FIR type for this part-ref.
  fir::FortranVariableOpInterface genDesignate(mlir::Type resultValueType,
                                               PartInfo &partInfo,
                                               const T &designatorNode) {
    mlir::Type designatorType =
        computeDesignatorType(resultValueType, partInfo, designatorNode);
    std::optional<bool> complexPart;
    auto designate = getBuilder().create<hlfir::DesignateOp>(
        getLoc(), designatorType, partInfo.base.getBase(),
        partInfo.componentName, partInfo.componentShape, partInfo.subscripts,
        partInfo.substring, complexPart, partInfo.resultShape,
        partInfo.typeParams);
    return mlir::cast<fir::FortranVariableOpInterface>(
        designate.getOperation());
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::SymbolRef &symbolRef) {
    if (std::optional<fir::FortranVariableOpInterface> varDef =
            getSymMap().lookupVariableDefinition(symbolRef))
      return *varDef;
    TODO(getLoc(), "lowering symbol to HLFIR");
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::Component &component) {
    PartInfo partInfo;
    mlir::Type resultType = visit(component, partInfo);
    return genDesignate(resultType, partInfo, component);
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    PartInfo partInfo;
    mlir::Type resultType = visit(arrayRef, partInfo);
    return genDesignate(resultType, partInfo, arrayRef);
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::CoarrayRef &coarrayRef) {
    TODO(getLoc(), "lowering CoarrayRef to HLFIR");
  }

  mlir::Type visit(const Fortran::evaluate::CoarrayRef &, PartInfo &) {
    TODO(getLoc(), "lowering CoarrayRef to HLFIR");
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::ComplexPart &complexPart) {
    TODO(getLoc(), "lowering complex part to HLFIR");
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::Substring &substring) {
    PartInfo partInfo;
    mlir::Type baseStringType = std::visit(
        [&](const auto &x) { return visit(x, partInfo); }, substring.parent());
    assert(partInfo.typeParams.size() == 1 && "expect base string length");
    // Compute the substring lower and upper bound.
    partInfo.substring.push_back(genSubscript(substring.lower()));
    if (Fortran::evaluate::MaybeExtentExpr upperBound = substring.upper())
      partInfo.substring.push_back(genSubscript(*upperBound));
    else
      partInfo.substring.push_back(partInfo.typeParams[0]);
    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    mlir::Type idxTy = builder.getIndexType();
    partInfo.substring[0] =
        builder.createConvert(loc, idxTy, partInfo.substring[0]);
    partInfo.substring[1] =
        builder.createConvert(loc, idxTy, partInfo.substring[1]);
    // Try using constant length if available. mlir::arith folding would
    // most likely be able to fold "max(ub-lb+1,0)" too, but getting
    // the constant length in the FIR types would be harder.
    std::optional<int64_t> cstLen =
        Fortran::evaluate::ToInt64(Fortran::evaluate::Fold(
            getConverter().getFoldingContext(), substring.LEN()));
    if (cstLen) {
      partInfo.typeParams[0] =
          builder.createIntegerConstant(loc, idxTy, *cstLen);
    } else {
      // Compute "len = max(ub-lb+1,0)" (Fortran 2018 9.4.1).
      mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
      auto boundsDiff = builder.create<mlir::arith::SubIOp>(
          loc, partInfo.substring[1], partInfo.substring[0]);
      auto rawLen = builder.create<mlir::arith::AddIOp>(loc, boundsDiff, one);
      partInfo.typeParams[0] =
          fir::factory::genMaxWithZero(builder, loc, rawLen);
    }
    auto kind = hlfir::getFortranElementType(baseStringType)
                    .cast<fir::CharacterType>()
                    .getFKind();
    auto newCharTy = fir::CharacterType::get(
        baseStringType.getContext(), kind,
        cstLen ? *cstLen : fir::CharacterType::unknownLen());
    mlir::Type resultType = changeElementType(baseStringType, newCharTy);
    return genDesignate(resultType, partInfo, substring);
  }

  static mlir::Type changeElementType(mlir::Type type, mlir::Type newEleTy) {
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
        .Case<fir::SequenceType>([&](fir::SequenceType seqTy) -> mlir::Type {
          return fir::SequenceType::get(seqTy.getShape(), newEleTy);
        })
        .Case<fir::PointerType, fir::HeapType, fir::ReferenceType,
              fir::BoxType>([&](auto t) -> mlir::Type {
          using FIRT = decltype(t);
          return FIRT::get(changeElementType(t.getEleTy(), newEleTy));
        })
        .Default([newEleTy](mlir::Type t) -> mlir::Type { return newEleTy; });
  }

  mlir::Type visit(const Fortran::evaluate::DataRef &dataRef,
                   PartInfo &partInfo) {
    return std::visit([&](const auto &x) { return visit(x, partInfo); },
                      dataRef.u);
  }

  mlir::Type
  visit(const Fortran::evaluate::StaticDataObject::Pointer &staticObject,
        PartInfo &partInfo) {
    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    std::optional<std::string> string = staticObject->AsString();
    // TODO: see if StaticDataObject can be replaced by something based on
    // Constant<T> to avoid dealing with endianness here for KIND>1.
    // This will also avoid making string copies here.
    if (!string)
      TODO(loc, "StaticDataObject::Pointer substring with kind > 1");
    fir::ExtendedValue exv =
        fir::factory::createStringLiteral(builder, getLoc(), *string);
    auto flags = fir::FortranVariableFlagsAttr::get(
        builder.getContext(), fir::FortranVariableFlagsEnum::parameter);
    partInfo.base = hlfir::genDeclare(loc, builder, exv, ".stringlit", flags);
    partInfo.typeParams.push_back(fir::getLen(exv));
    return partInfo.base.getElementOrSequenceType();
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
      baseType = visitComponentImpl(*component, partInfo).second;
    else
      baseType = visit(arrayRef.base().GetLastSymbol(), partInfo);

    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    mlir::Type idxTy = builder.getIndexType();
    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> bounds;
    auto getBaseBounds = [&](unsigned i) {
      if (bounds.empty()) {
        if (partInfo.componentName.empty()) {
          bounds = hlfir::genBounds(loc, builder, partInfo.base);
        } else {
          assert(
              partInfo.componentShape &&
              "implicit array section bounds must come from component shape");
          bounds = hlfir::genBounds(loc, builder, partInfo.componentShape);
        }
        assert(!bounds.empty() &&
               "failed to compute implicit array section bounds");
      }
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
          lb = getBaseBounds(subscript.index()).first;
        if (const auto &ubExpr = triplet->upper())
          ub = genSubscript(*ubExpr);
        else
          ub = getBaseBounds(subscript.index()).second;
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
      // Ranked array section. The result shape comes from the array section
      // subscripts.
      resultType = fir::SequenceType::get(resultTypeShape, resultType);
      assert(!partInfo.resultShape &&
             "Fortran designator can only have one ranked part");
      partInfo.resultShape = builder.genShape(loc, resultExtents);
    } else if (!partInfo.componentName.empty() && partInfo.base.isArray()) {
      // This is an array%array_comp(indices) reference. Keep the
      // shape of the base array and not the array_comp.
      auto compBaseTy = partInfo.base.getElementOrSequenceType();
      resultType = changeElementType(compBaseTy, resultType);
      assert(!partInfo.resultShape && "should not have been computed already");
      partInfo.resultShape = hlfir::genShape(loc, builder, partInfo.base);
    }
    return resultType;
  }

  static bool
  hasNonDefaultLowerBounds(const Fortran::semantics::Symbol &componentSym) {
    if (const auto *objDetails =
            componentSym.detailsIf<Fortran::semantics::ObjectEntityDetails>())
      for (const Fortran::semantics::ShapeSpec &bounds : objDetails->shape())
        if (auto lb = bounds.lbound().GetExplicit())
          if (auto constant = Fortran::evaluate::ToInt64(*lb))
            if (!constant || *constant != 1)
              return true;
    return false;
  }
  static bool hasNonDefaultLowerBounds(const PartInfo &partInfo) {
    return partInfo.resultShape &&
           (partInfo.resultShape.getType().isa<fir::ShiftType>() ||
            partInfo.resultShape.getType().isa<fir::ShapeShiftType>());
  }

  mlir::Value genComponentShape(const Fortran::semantics::Symbol &componentSym,
                                mlir::Type fieldType) {
    // For pointers and allocatable components, the
    // shape is deferred and should not be loaded now to preserve
    // pointer/allocatable aspects.
    if (componentSym.Rank() == 0 ||
        Fortran::semantics::IsAllocatableOrPointer(componentSym))
      return mlir::Value{};

    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    mlir::Type idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> extents;
    auto seqTy = hlfir::getFortranElementOrSequenceType(fieldType)
                     .cast<fir::SequenceType>();
    for (auto extent : seqTy.getShape())
      extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
    if (!hasNonDefaultLowerBounds(componentSym))
      return builder.create<fir::ShapeOp>(loc, extents);

    llvm::SmallVector<mlir::Value> lbounds;
    if (const auto *objDetails =
            componentSym.detailsIf<Fortran::semantics::ObjectEntityDetails>())
      for (const Fortran::semantics::ShapeSpec &bounds : objDetails->shape())
        if (auto lb = bounds.lbound().GetExplicit())
          if (auto constant = Fortran::evaluate::ToInt64(*lb))
            lbounds.push_back(
                builder.createIntegerConstant(loc, idxTy, *constant));
    assert(extents.size() == lbounds.size() &&
           "extents and lower bounds must match");
    return builder.genShape(loc, lbounds, extents);
  }

  mlir::Type visit(const Fortran::evaluate::Component &component,
                   PartInfo &partInfo) {
    // Called from contexts where the component is not the base of an ArrayRef.
    // In these cases, the component cannot be an array if the base is an
    // array. The code below determines the shape of the component reference if
    // any.
    auto [baseType, componentType] = visitComponentImpl(component, partInfo);
    if (partInfo.base.isArray()) {
      // For array%scalar_comp, the result shape is
      // the one of the base. Compute it here. Note that the lower bounds of the
      // base are not the ones of the resulting reference (that are default
      // ones).
      partInfo.resultShape = hlfir::genShape(loc, getBuilder(), partInfo.base);
      assert(!partInfo.componentShape &&
             "Fortran designators can only have one ranked part");
      return changeElementType(baseType, componentType);
    }
    // scalar%array_comp or scalar%scalar. In any case the shape of this
    // part-ref is coming from the component.
    partInfo.resultShape = partInfo.componentShape;
    partInfo.componentShape = {};
    return componentType;
  }

  // Returns the <BaseType, ComponentType> pair, computes partInfo.base,
  // partInfo.componentShape and partInfo.typeParams, but does not set the
  // partInfo.resultShape yet. The result shape will be computed after
  // processing a following ArrayRef, if any, and in "visit" otherwise.
  std::pair<mlir::Type, mlir::Type>
  visitComponentImpl(const Fortran::evaluate::Component &component,
                     PartInfo &partInfo) {
    fir::FirOpBuilder &builder = getBuilder();
    // Break the Designator visit here: if the base is an array-ref, a
    // coarray-ref, or another component, this creates another hlfir.designate
    // for it.  hlfir.designate is not meant to represent more than one
    // part-ref.
    partInfo.base =
        std::visit([&](const auto &x) { return gen(x); }, component.base().u);
    assert(partInfo.typeParams.empty() && "should not have been computed yet");
    hlfir::genLengthParameters(getLoc(), getBuilder(), partInfo.base,
                               partInfo.typeParams);
    mlir::Type baseType = partInfo.base.getElementOrSequenceType();

    // Lower the information about the component (type, length parameters and
    // shape).
    const Fortran::semantics::Symbol &componentSym = component.GetLastSymbol();
    partInfo.componentName = componentSym.name().ToString();
    auto recordType =
        hlfir::getFortranElementType(baseType).cast<fir::RecordType>();
    if (recordType.isDependentType())
      TODO(getLoc(), "Designate derived type with length parameters in HLFIR");
    mlir::Type fieldType = recordType.getType(partInfo.componentName);
    fieldType = hlfir::getFortranElementOrSequenceType(fieldType);
    partInfo.componentShape = genComponentShape(componentSym, fieldType);

    mlir::Type fieldEleType = hlfir::getFortranElementType(fieldType);
    if (fir::isRecordWithTypeParameters(fieldEleType))
      TODO(loc,
           "lower a component that is a parameterized derived type to HLFIR");
    if (auto charTy = fieldEleType.dyn_cast<fir::CharacterType>()) {
      mlir::Location loc = getLoc();
      mlir::Type idxTy = builder.getIndexType();
      if (charTy.hasConstantLen())
        partInfo.typeParams.push_back(
            builder.createIntegerConstant(loc, idxTy, charTy.getLen()));
      else if (!Fortran::semantics::IsAllocatableOrPointer(componentSym))
        TODO(loc, "compute character length of automatic character component "
                  "in a PDT");
      // Otherwise, the length of the component is deferred and will only
      // be read when the component is dereferenced.
    }

    // For pointers and allocatables, if there is a substring, complex part or
    // array ref, the designator should be broken here and the pointer or
    // allocatable dereferenced.
    if (Fortran::semantics::IsAllocatableOrPointer(componentSym))
      TODO(loc, "lowering ref to allocatable or pointer component to HLFIR");

    return {baseType, fieldType};
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
    llvm::SmallVector<mlir::Value, 2> args{lhs, rhs};
    fir::ExtendedValue res = op.ordering == Fortran::evaluate::Ordering::Greater
                                 ? Fortran::lower::genMax(builder, loc, args)
                                 : Fortran::lower::genMin(builder, loc, args);
    return hlfir::EntityWithAttributes{fir::getBase(res)};
  }
};

// evaluate::Extremum is only created by the front-end when building compiler
// generated expressions (like when folding LEN() or shape/bounds inquiries).
// MIN and MAX are represented as evaluate::ProcedureRef and are not going
// through here. So far the frontend does not generate character Extremum so
// there is no way to test it.
template <int KIND>
struct BinaryOp<Fortran::evaluate::Extremum<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>> {
  using Op = Fortran::evaluate::Extremum<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &, const Op &,
                                         hlfir::Entity, hlfir::Entity) {
    fir::emitFatalError(loc, "Fortran::evaluate::Extremum are unexpected");
  }
  static void genResultTypeParams(mlir::Location loc, fir::FirOpBuilder &,
                                  hlfir::Entity, hlfir::Entity,
                                  llvm::SmallVectorImpl<mlir::Value> &) {
    fir::emitFatalError(loc, "Fortran::evaluate::Extremum are unexpected");
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
      (*lhsCleanUp)();
    if (rhsCleanUp)
      (*rhsCleanUp)();
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
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity string,
                                         hlfir::Entity length) {
    return hlfir::EntityWithAttributes{
        builder.create<hlfir::SetLengthOp>(loc, string, length)};
  }
  static void
  genResultTypeParams(mlir::Location, fir::FirOpBuilder &, hlfir::Entity,
                      hlfir::Entity rhs,
                      llvm::SmallVectorImpl<mlir::Value> &resultTypeParams) {
    resultTypeParams.push_back(rhs);
  }
};

template <int KIND>
struct BinaryOp<Fortran::evaluate::Concat<KIND>> {
  using Op = Fortran::evaluate::Concat<KIND>;
  hlfir::EntityWithAttributes gen(mlir::Location loc,
                                  fir::FirOpBuilder &builder, const Op &,
                                  hlfir::Entity lhs, hlfir::Entity rhs) {
    assert(len && "genResultTypeParams must have been called");
    auto concat =
        builder.create<hlfir::ConcatOp>(loc, mlir::ValueRange{lhs, rhs}, len);
    return hlfir::EntityWithAttributes{concat.getResult()};
  }
  void
  genResultTypeParams(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity lhs, hlfir::Entity rhs,
                      llvm::SmallVectorImpl<mlir::Value> &resultTypeParams) {
    llvm::SmallVector<mlir::Value> lengths;
    hlfir::genLengthParameters(loc, builder, lhs, lengths);
    hlfir::genLengthParameters(loc, builder, rhs, lengths);
    assert(lengths.size() == 2 && "lacks rhs or lhs length");
    mlir::Type idxType = builder.getIndexType();
    mlir::Value lhsLen = builder.createConvert(loc, idxType, lengths[0]);
    mlir::Value rhsLen = builder.createConvert(loc, idxType, lengths[1]);
    len = builder.create<mlir::arith::AddIOp>(loc, lhsLen, rhsLen);
    resultTypeParams.push_back(len);
  }

private:
  mlir::Value len{};
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

  static void
  genResultTypeParams(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity lhs,
                      llvm::SmallVectorImpl<mlir::Value> &resultTypeParams) {
    hlfir::genLengthParameters(loc, builder, lhs, resultTypeParams);
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

  static void
  genResultTypeParams(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity lhs,
                      llvm::SmallVectorImpl<mlir::Value> &resultTypeParams) {
    hlfir::genLengthParameters(loc, builder, lhs, resultTypeParams);
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
    auto nullop = getBuilder().create<hlfir::NullOp>(getLoc());
    return mlir::cast<fir::FortranVariableOpInterface>(nullop.getOperation());
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
    auto result = Fortran::lower::convertCallToHLFIR(
        getLoc(), getConverter(), expr, resType, getSymMap(), getStmtCtx());
    assert(result.has_value());
    return *result;
  }

  template <typename T>
  hlfir::EntityWithAttributes gen(const Fortran::evaluate::Constant<T> &expr) {
    mlir::Location loc = getLoc();
    fir::FirOpBuilder &builder = getBuilder();
    fir::ExtendedValue exv = Fortran::lower::convertConstant(
        converter, loc, expr, /*outlineBigConstantInReadOnlyMemory=*/true);
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
    const int rank = op.Rank();
    UnaryOp<D> unaryOp;
    auto left = hlfir::loadTrivialScalar(loc, builder, gen(op.left()));
    llvm::SmallVector<mlir::Value, 1> typeParams;
    if constexpr (R::category == Fortran::common::TypeCategory::Character) {
      unaryOp.genResultTypeParams(loc, builder, left, typeParams);
    }
    if (rank == 0)
      return unaryOp.gen(loc, builder, op.derived(), left);

    // Elemental expression.
    mlir::Type elementType;
    if constexpr (R::category == Fortran::common::TypeCategory::Derived) {
      elementType = Fortran::lower::translateDerivedTypeToFIRType(
          getConverter(), op.derived().GetType().GetDerivedTypeSpec());
    } else {
      elementType =
          Fortran::lower::getFIRType(builder.getContext(), R::category, R::kind,
                                     /*params=*/std::nullopt);
    }
    mlir::Value shape = hlfir::genShape(loc, builder, left);
    auto genKernel = [&op, &left, &unaryOp](
                         mlir::Location l, fir::FirOpBuilder &b,
                         mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
      auto leftElement = hlfir::getElementAt(l, b, left, oneBasedIndices);
      auto leftVal = hlfir::loadTrivialScalar(l, b, leftElement);
      return unaryOp.gen(l, b, op.derived(), leftVal);
    };
    mlir::Value elemental = hlfir::genElementalOp(loc, builder, elementType,
                                                  shape, typeParams, genKernel);
    fir::FirOpBuilder *bldr = &builder;
    getStmtCtx().attachCleanup(
        [=]() { bldr->create<hlfir::DestroyOp>(loc, elemental); });
    return hlfir::EntityWithAttributes{elemental};
  }

  template <typename D, typename R, typename LO, typename RO>
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    auto &builder = getBuilder();
    mlir::Location loc = getLoc();
    const int rank = op.Rank();
    BinaryOp<D> binaryOp;
    auto left = hlfir::loadTrivialScalar(loc, builder, gen(op.left()));
    auto right = hlfir::loadTrivialScalar(loc, builder, gen(op.right()));
    llvm::SmallVector<mlir::Value, 1> typeParams;
    if constexpr (R::category == Fortran::common::TypeCategory::Character) {
      binaryOp.genResultTypeParams(loc, builder, left, right, typeParams);
    }
    if (rank == 0)
      return binaryOp.gen(loc, builder, op.derived(), left, right);

    // Elemental expression.
    mlir::Type elementType =
        Fortran::lower::getFIRType(builder.getContext(), R::category, R::kind,
                                   /*params=*/std::nullopt);
    // TODO: "merge" shape, get cst shape from front-end if possible.
    mlir::Value shape;
    if (left.isArray()) {
      shape = hlfir::genShape(loc, builder, left);
    } else {
      assert(right.isArray() && "must have at least one array operand");
      shape = hlfir::genShape(loc, builder, right);
    }
    auto genKernel = [&op, &left, &right, &binaryOp](
                         mlir::Location l, fir::FirOpBuilder &b,
                         mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
      auto leftElement = hlfir::getElementAt(l, b, left, oneBasedIndices);
      auto rightElement = hlfir::getElementAt(l, b, right, oneBasedIndices);
      auto leftVal = hlfir::loadTrivialScalar(l, b, leftElement);
      auto rightVal = hlfir::loadTrivialScalar(l, b, rightElement);
      return binaryOp.gen(l, b, op.derived(), leftVal, rightVal);
    };
    mlir::Value elemental = hlfir::genElementalOp(loc, builder, elementType,
                                                  shape, typeParams, genKernel);
    fir::FirOpBuilder *bldr = &builder;
    getStmtCtx().attachCleanup(
        [=]() { bldr->create<hlfir::DestroyOp>(loc, elemental); });
    return hlfir::EntityWithAttributes{elemental};
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
    mlir::Location loc = getLoc();
    auto &builder = getBuilder();
    hlfir::EntityWithAttributes entity =
        HlfirDesignatorBuilder(getLoc(), getConverter(), getSymMap(),
                               getStmtCtx())
            .gen(desc.base());
    using ResTy = Fortran::evaluate::DescriptorInquiry::Result;
    mlir::Type resultType =
        getConverter().genType(ResTy::category, ResTy::kind);
    auto castResult = [&](mlir::Value v) {
      return hlfir::EntityWithAttributes{
          builder.createConvert(loc, resultType, v)};
    };
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      return castResult(hlfir::genCharLength(loc, builder, entity));
    case Fortran::evaluate::DescriptorInquiry::Field::LowerBound:
      TODO(loc, "lower bound inquiry in HLFIR");
    case Fortran::evaluate::DescriptorInquiry::Field::Extent:
      TODO(loc, "extent inquiry in HLFIR");
    case Fortran::evaluate::DescriptorInquiry::Field::Rank:
      TODO(loc, "rank inquiry on assumed rank");
    case Fortran::evaluate::DescriptorInquiry::Field::Stride:
      // So far the front end does not generate this inquiry.
      TODO(loc, "stride inquiry");
    }
    llvm_unreachable("unknown descriptor inquiry");
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
      if (auto cstIndex = fir::getIntIfConstant(loweredExpr))
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

fir::BoxValue Fortran::lower::convertToBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    hlfir::Entity entity, Fortran::lower::StatementContext &stmtCtx) {
  auto exv = Fortran::lower::translateToExtendedValue(
      loc, converter.getFirOpBuilder(), entity, stmtCtx);
  if (fir::isa_trivial(fir::getBase(exv).getType()))
    TODO(loc, "place trivial in memory");
  return fir::factory::createBoxValue(converter.getFirOpBuilder(), loc, exv);
}
fir::BoxValue Fortran::lower::convertExprToBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  hlfir::EntityWithAttributes loweredExpr =
      HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
  return convertToBox(loc, converter, loweredExpr, stmtCtx);
}

fir::ExtendedValue
Fortran::lower::convertToAddress(mlir::Location loc,
                                 Fortran::lower::AbstractConverter &converter,
                                 hlfir::Entity entity, bool isSimplyContiguous,
                                 Fortran::lower::StatementContext &stmtCtx) {
  if (!isSimplyContiguous)
    TODO(loc, "genExprAddr of non contiguous variables in HLFIR");
  fir::ExtendedValue exv = Fortran::lower::translateToExtendedValue(
      loc, converter.getFirOpBuilder(), entity, stmtCtx);
  if (fir::isa_trivial(fir::getBase(exv).getType()))
    TODO(loc, "place trivial in memory");
  if (const auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>())
    exv = fir::factory::genMutableBoxRead(converter.getFirOpBuilder(), loc,
                                          *mutableBox);
  return exv;
}
fir::ExtendedValue Fortran::lower::convertExprToAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  hlfir::EntityWithAttributes loweredExpr =
      HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
  bool isSimplyContiguous =
      expr.Rank() == 0 || Fortran::evaluate::IsSimplyContiguous(
                              expr, converter.getFoldingContext());
  return convertToAddress(loc, converter, loweredExpr, isSimplyContiguous,
                          stmtCtx);
}

fir::ExtendedValue Fortran::lower::convertToValue(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    hlfir::Entity entity, Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  fir::ExtendedValue exv =
      Fortran::lower::translateToExtendedValue(loc, builder, entity, stmtCtx);
  // Load scalar references to integer, logical, real, or complex value
  // to an mlir value, dereference allocatable and pointers, and get rid
  // of fir.box that are not needed or create a copy into contiguous memory.
  return exv.match(
      [&](const fir::UnboxedValue &box) -> fir::ExtendedValue {
        if (mlir::Type elementType = fir::dyn_cast_ptrEleTy(box.getType()))
          if (fir::isa_trivial(elementType))
            return builder.create<fir::LoadOp>(loc, box);
        return box;
      },
      [&](const fir::CharBoxValue &box) -> fir::ExtendedValue { return box; },
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue { return box; },
      [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        TODO(loc, "lower descriptor designator to HLFIR value");
      });
}

fir::ExtendedValue Fortran::lower::convertExprToValue(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  hlfir::EntityWithAttributes loweredExpr =
      HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
  return convertToValue(loc, converter, loweredExpr, stmtCtx);
}
