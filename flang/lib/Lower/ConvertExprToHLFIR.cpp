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
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertArrayConstructor.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertConstant.h"
#include "flang/Lower/ConvertProcedureDesignator.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/Pointer.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

namespace {

/// Lower Designators to HLFIR.
class HlfirDesignatorBuilder {
private:
  /// Internal entry point on the rightest part of a evaluate::Designator.
  template <typename T>
  hlfir::EntityWithAttributes
  genLeafPartRef(const T &designatorNode,
                 bool vectorSubscriptDesignatorToValue) {
    hlfir::EntityWithAttributes result = gen(designatorNode);
    if (vectorSubscriptDesignatorToValue)
      return turnVectorSubscriptedDesignatorIntoValue(result);
    return result;
  }

  hlfir::EntityWithAttributes
  genDesignatorExpr(const Fortran::lower::SomeExpr &designatorExpr,
                    bool vectorSubscriptDesignatorToValue = true);

public:
  HlfirDesignatorBuilder(mlir::Location loc,
                         Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::SymMap &symMap,
                         Fortran::lower::StatementContext &stmtCtx)
      : converter{converter}, symMap{symMap}, stmtCtx{stmtCtx}, loc{loc} {}

  /// Public entry points to lower a Designator<T> (given its .u member, to
  /// avoid the template arguments which does not matter here).
  /// This lowers a designator to an hlfir variable SSA value (that can be
  /// assigned to), except for vector subscripted designators that are
  /// lowered by default to hlfir.expr value since they cannot be
  /// represented as HLFIR variable SSA values.

  // Character designators variant contains substrings
  using CharacterDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Character, 1>>::u);
  hlfir::EntityWithAttributes
  gen(const CharacterDesignators &designatorVariant,
      bool vectorSubscriptDesignatorToValue = true) {
    return std::visit(
        [&](const auto &x) -> hlfir::EntityWithAttributes {
          return genLeafPartRef(x, vectorSubscriptDesignatorToValue);
        },
        designatorVariant);
  }
  // Character designators variant contains complex parts
  using RealDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Real, 4>>::u);
  hlfir::EntityWithAttributes
  gen(const RealDesignators &designatorVariant,
      bool vectorSubscriptDesignatorToValue = true) {
    return std::visit(
        [&](const auto &x) -> hlfir::EntityWithAttributes {
          return genLeafPartRef(x, vectorSubscriptDesignatorToValue);
        },
        designatorVariant);
  }
  // All other designators are similar
  using OtherDesignators =
      decltype(Fortran::evaluate::Designator<Fortran::evaluate::Type<
                   Fortran::evaluate::TypeCategory::Integer, 4>>::u);
  hlfir::EntityWithAttributes
  gen(const OtherDesignators &designatorVariant,
      bool vectorSubscriptDesignatorToValue = true) {
    return std::visit(
        [&](const auto &x) -> hlfir::EntityWithAttributes {
          return genLeafPartRef(x, vectorSubscriptDesignatorToValue);
        },
        designatorVariant);
  }

  hlfir::EntityWithAttributes
  genNamedEntity(const Fortran::evaluate::NamedEntity &namedEntity,
                 bool vectorSubscriptDesignatorToValue = true) {
    if (namedEntity.IsSymbol())
      return genLeafPartRef(
          Fortran::evaluate::SymbolRef{namedEntity.GetLastSymbol()},
          vectorSubscriptDesignatorToValue);
    return genLeafPartRef(namedEntity.GetComponent(),
                          vectorSubscriptDesignatorToValue);
  }

  /// Public entry point to lower a vector subscripted designator to
  /// an hlfir::ElementalAddrOp.
  hlfir::ElementalAddrOp convertVectorSubscriptedExprToElementalAddr(
      const Fortran::lower::SomeExpr &designatorExpr);

  mlir::Value genComponentShape(const Fortran::semantics::Symbol &componentSym,
                                mlir::Type fieldType) {
    // For pointers and allocatable components, the
    // shape is deferred and should not be loaded now to preserve
    // pointer/allocatable aspects.
    if (componentSym.Rank() == 0 ||
        Fortran::semantics::IsAllocatableOrObjectPointer(&componentSym))
      return mlir::Value{};

    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    mlir::Type idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> extents;
    auto seqTy = hlfir::getFortranElementOrSequenceType(fieldType)
                     .cast<fir::SequenceType>();
    for (auto extent : seqTy.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent()) {
        // We have already generated invalid hlfir.declare
        // without the type parameters and probably invalid storage
        // for the variable (e.g. fir.alloca without type parameters).
        // So this TODO here is a little bit late, but it matches
        // the non-HLFIR path.
        TODO(loc, "array component shape depending on length parameters");
      }
      extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
    }
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

private:
  /// Struct that is filled while visiting a part-ref (in the "visit" member
  /// function) before the top level "gen" generates an hlfir.declare for the
  /// part ref. It contains the lowered pieces of the part-ref that will
  /// become the operands of an hlfir.declare.
  struct PartInfo {
    std::optional<hlfir::Entity> base;
    std::string componentName{};
    mlir::Value componentShape;
    hlfir::DesignateOp::Subscripts subscripts;
    std::optional<bool> complexPart;
    mlir::Value resultShape;
    llvm::SmallVector<mlir::Value> typeParams;
    llvm::SmallVector<mlir::Value, 2> substring;
  };

  // Given the value type of a designator (T or fir.array<T>) and the front-end
  // node for the designator, compute the memory type (fir.class, fir.ref, or
  // fir.box)...
  template <typename T>
  mlir::Type computeDesignatorType(mlir::Type resultValueType,
                                   PartInfo &partInfo,
                                   const T &designatorNode) {
    // Get base's shape if its a sequence type with no previously computed
    // result shape
    if (partInfo.base && resultValueType.isa<fir::SequenceType>() &&
        !partInfo.resultShape)
      partInfo.resultShape =
          hlfir::genShape(getLoc(), getBuilder(), *partInfo.base);
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
    return genDesignate(designatorType, partInfo, /*attributes=*/{});
  }
  fir::FortranVariableOpInterface
  genDesignate(mlir::Type designatorType, PartInfo &partInfo,
               fir::FortranVariableFlagsAttr attributes) {
    fir::FirOpBuilder &builder = getBuilder();
    // Once a part with vector subscripts has been lowered, the following
    // hlfir.designator (for the parts on the right of the designator) must
    // be lowered inside the hlfir.elemental_addr because they depend on the
    // hlfir.elemental_addr indices.
    // All the subsequent Fortran indices however, should be lowered before
    // the hlfir.elemental_addr because they should only be evaluated once,
    // hence, the insertion point is restored outside of the
    // hlfir.elemental_addr after generating the hlfir.designate. Example: in
    // "X(VECTOR)%COMP(FOO(), BAR())", the calls to bar() and foo() must be
    // generated outside of the hlfir.elemental, but the related hlfir.designate
    // that depends on the scalar hlfir.designate of X(VECTOR) that was
    // generated inside the hlfir.elemental_addr should be generated in the
    // hlfir.elemental_addr.
    if (auto elementalAddrOp = getVectorSubscriptElementAddrOp())
      builder.setInsertionPointToEnd(&elementalAddrOp->getBody().front());
    auto designate = builder.create<hlfir::DesignateOp>(
        getLoc(), designatorType, partInfo.base.value().getBase(),
        partInfo.componentName, partInfo.componentShape, partInfo.subscripts,
        partInfo.substring, partInfo.complexPart, partInfo.resultShape,
        partInfo.typeParams, attributes);
    if (auto elementalAddrOp = getVectorSubscriptElementAddrOp())
      builder.setInsertionPoint(*elementalAddrOp);
    return mlir::cast<fir::FortranVariableOpInterface>(
        designate.getOperation());
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::SymbolRef &symbolRef) {
    if (std::optional<fir::FortranVariableOpInterface> varDef =
            getSymMap().lookupVariableDefinition(symbolRef)) {
      if (symbolRef->test(Fortran::semantics::Symbol::Flag::CrayPointee)) {
        // The pointee is represented with a descriptor inheriting
        // the shape and type parameters of the pointee.
        // We have to update the base_addr to point to the current
        // value of the Cray pointer variable.
        fir::FirOpBuilder &builder = getBuilder();
        fir::FortranVariableOpInterface ptrVar =
            gen(Fortran::lower::getCrayPointer(symbolRef));
        mlir::Value ptrAddr = ptrVar.getBase();

        // Reinterpret the reference to a Cray pointer so that
        // we have a pointer-compatible value after loading
        // the Cray pointer value.
        mlir::Type refPtrType = builder.getRefType(
            fir::PointerType::get(fir::dyn_cast_ptrEleTy(ptrAddr.getType())));
        mlir::Value cast = builder.createConvert(loc, refPtrType, ptrAddr);
        mlir::Value ptrVal = builder.create<fir::LoadOp>(loc, cast);

        // Update the base_addr to the value of the Cray pointer.
        // This is a hacky way to do the update, and it may harm
        // performance around Cray pointer references.
        // TODO: we should introduce an operation that updates
        // just the base_addr of the given box. The CodeGen
        // will just convert it into a single store.
        fir::runtime::genPointerAssociateScalar(builder, loc, varDef->getBase(),
                                                ptrVal);
      }
      return *varDef;
    }
    TODO(getLoc(), "lowering symbol to HLFIR");
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::Component &component,
      bool skipParentComponent = false) {
    if (Fortran::semantics::IsAllocatableOrPointer(component.GetLastSymbol()))
      return genWholeAllocatableOrPointerComponent(component);
    if (component.GetLastSymbol().test(
            Fortran::semantics::Symbol::Flag::ParentComp)) {
      if (skipParentComponent)
        // Inner parent components can be skipped: x%parent_comp%i is equivalent
        // to "x%i" in FIR (all the parent components are part of the FIR type
        // of "x").
        return genDataRefAndSkipParentComponents(component.base());
      // This is a leaf "x%parent_comp" or "x(subscripts)%parent_comp" and
      // cannot be skipped: the designator must be lowered to the parent type.
      // This cannot be represented with an hlfir.designate since "parent_comp"
      // name is meaningless in the fir.record type of "x". Instead, an
      // hlfir.parent_comp is generated.
      fir::FirOpBuilder &builder = getBuilder();
      hlfir::Entity base = genDataRefAndSkipParentComponents(component.base());
      base = derefPointersAndAllocatables(loc, builder, base);
      mlir::Value shape;
      if (base.isArray())
        shape = hlfir::genShape(loc, builder, base);
      const Fortran::semantics::DeclTypeSpec *declTypeSpec =
          component.GetLastSymbol().GetType();
      assert(declTypeSpec && declTypeSpec->AsDerived() &&
             "parent component symbols must have a derived type");
      mlir::Type componentType = Fortran::lower::translateDerivedTypeToFIRType(
          getConverter(), *declTypeSpec->AsDerived());
      mlir::Type resultType =
          changeElementType(base.getElementOrSequenceType(), componentType);
      // Note that the result is monomorphic even if the base is polymorphic:
      // the dynamic type of the parent component reference is the parent type.
      // If the base is an array, it is however most likely not contiguous.
      if (base.isArray() || fir::isRecordWithTypeParameters(componentType))
        resultType = fir::BoxType::get(resultType);
      else
        resultType = fir::ReferenceType::get(resultType);
      if (fir::isRecordWithTypeParameters(componentType))
        TODO(loc, "parent component reference with a parametrized parent type");
      auto parentComp = builder.create<hlfir::ParentComponentOp>(
          loc, resultType, base, shape, /*typeParams=*/mlir::ValueRange{});
      return mlir::cast<fir::FortranVariableOpInterface>(
          parentComp.getOperation());
    }
    PartInfo partInfo;
    mlir::Type resultType = visit(component, partInfo);
    return genDesignate(resultType, partInfo, component);
  }

  fir::FortranVariableOpInterface
  genDataRefAndSkipParentComponents(const Fortran::evaluate::DataRef &dataRef) {
    return std::visit(Fortran::common::visitors{
                          [&](const Fortran::evaluate::Component &component) {
                            return gen(component, /*skipParentComponent=*/true);
                          },
                          [&](const auto &x) { return gen(x); }},
                      dataRef.u);
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    PartInfo partInfo;
    mlir::Type resultType = visit(arrayRef, partInfo);
    return genDesignate(resultType, partInfo, arrayRef);
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::CoarrayRef &coarrayRef) {
    TODO(getLoc(), "coarray: lowering a reference to a coarray object");
  }

  mlir::Type visit(const Fortran::evaluate::CoarrayRef &, PartInfo &) {
    TODO(getLoc(), "coarray: lowering a reference to a coarray object");
  }

  fir::FortranVariableOpInterface
  gen(const Fortran::evaluate::ComplexPart &complexPart) {
    PartInfo partInfo;
    fir::factory::Complex cmplxHelper(getBuilder(), getLoc());

    bool complexBit =
        complexPart.part() == Fortran::evaluate::ComplexPart::Part::IM;
    partInfo.complexPart = {complexBit};

    mlir::Type resultType = visit(complexPart.complex(), partInfo);

    // Determine complex part type
    mlir::Type base = hlfir::getFortranElementType(resultType);
    mlir::Type cmplxValueType = cmplxHelper.getComplexPartType(base);
    mlir::Type designatorType = changeElementType(resultType, cmplxValueType);

    return genDesignate(designatorType, partInfo, complexPart);
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

  fir::FortranVariableOpInterface genWholeAllocatableOrPointerComponent(
      const Fortran::evaluate::Component &component) {
    // Generate whole allocatable or pointer component reference. The
    // hlfir.designate result will be a pointer/allocatable.
    PartInfo partInfo;
    mlir::Type componentType = visitComponentImpl(component, partInfo).second;
    mlir::Type designatorType = fir::ReferenceType::get(componentType);
    fir::FortranVariableFlagsAttr attributes =
        Fortran::lower::translateSymbolAttributes(getBuilder().getContext(),
                                                  component.GetLastSymbol());
    return genDesignate(designatorType, partInfo, attributes);
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
    return partInfo.base->getElementOrSequenceType();
  }

  mlir::Type visit(const Fortran::evaluate::SymbolRef &symbolRef,
                   PartInfo &partInfo) {
    // A symbol is only visited if there is a following array, substring, or
    // complex reference. If the entity is a pointer or allocatable, this
    // reference designates the target, so the pointer, allocatable must be
    // dereferenced here.
    partInfo.base =
        hlfir::derefPointersAndAllocatables(loc, getBuilder(), gen(symbolRef));
    hlfir::genLengthParameters(loc, getBuilder(), *partInfo.base,
                               partInfo.typeParams);
    return partInfo.base->getElementOrSequenceType();
  }

  mlir::Type visit(const Fortran::evaluate::ArrayRef &arrayRef,
                   PartInfo &partInfo) {
    mlir::Type baseType;
    if (const auto *component = arrayRef.base().UnwrapComponent()) {
      // Pointers and allocatable components must be dereferenced since the
      // array ref designates the target (this is done in "visit"). Other
      // components need special care to deal with the array%array_comp(indices)
      // case.
      if (Fortran::semantics::IsAllocatableOrObjectPointer(
              &component->GetLastSymbol()))
        baseType = visit(*component, partInfo);
      else
        baseType = hlfir::getFortranElementOrSequenceType(
            visitComponentImpl(*component, partInfo).second);
    } else {
      baseType = visit(arrayRef.base().GetLastSymbol(), partInfo);
    }

    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    mlir::Type idxTy = builder.getIndexType();
    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> bounds;
    auto getBaseBounds = [&](unsigned i) {
      if (bounds.empty()) {
        if (partInfo.componentName.empty()) {
          bounds = hlfir::genBounds(loc, builder, partInfo.base.value());
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
    auto tryGettingExtentFromFrontEnd =
        [&](unsigned dim) -> std::pair<mlir::Value, fir::SequenceType::Extent> {
      // Use constant extent if possible. The main advantage to do this now
      // is to get the best FIR array types as possible while lowering.
      if (frontEndResultShape)
        if (auto maybeI64 =
                Fortran::evaluate::ToInt64(frontEndResultShape->at(dim)))
          return {builder.createIntegerConstant(loc, idxTy, *maybeI64),
                  *maybeI64};
      return {mlir::Value{}, fir::SequenceType::getUnknownExtent()};
    };
    llvm::SmallVector<mlir::Value> resultExtents;
    fir::SequenceType::Shape resultTypeShape;
    bool sawVectorSubscripts = false;
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
        auto [extentValue, shapeExtent] =
            tryGettingExtentFromFrontEnd(resultExtents.size());
        resultTypeShape.push_back(shapeExtent);
        if (!extentValue)
          extentValue =
              builder.genExtentFromTriplet(loc, lb, ub, stride, idxTy);
        resultExtents.push_back(extentValue);
        partInfo.subscripts.emplace_back(
            hlfir::DesignateOp::Triplet{lb, ub, stride});
      } else {
        const auto &expr =
            std::get<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
                subscript.value().u)
                .value();
        hlfir::Entity subscript = genSubscript(expr);
        partInfo.subscripts.push_back(subscript);
        if (expr.Rank() > 0) {
          sawVectorSubscripts = true;
          auto [extentValue, shapeExtent] =
              tryGettingExtentFromFrontEnd(resultExtents.size());
          resultTypeShape.push_back(shapeExtent);
          if (!extentValue)
            extentValue = hlfir::genExtent(loc, builder, subscript, /*dim=*/0);
          resultExtents.push_back(extentValue);
        }
      }
    }
    assert(resultExtents.size() == resultTypeShape.size() &&
           "inconsistent hlfir.designate shape");

    // For vector subscripts, create an hlfir.elemental_addr and continue
    // lowering the designator inside it as if it was addressing an element of
    // the vector subscripts.
    if (sawVectorSubscripts)
      return createVectorSubscriptElementAddrOp(partInfo, baseType,
                                                resultExtents);

    mlir::Type resultType = baseType.cast<fir::SequenceType>().getEleTy();
    if (!resultTypeShape.empty()) {
      // Ranked array section. The result shape comes from the array section
      // subscripts.
      resultType = fir::SequenceType::get(resultTypeShape, resultType);
      assert(!partInfo.resultShape &&
             "Fortran designator can only have one ranked part");
      partInfo.resultShape = builder.genShape(loc, resultExtents);
    } else if (!partInfo.componentName.empty() &&
               partInfo.base.value().isArray()) {
      // This is an array%array_comp(indices) reference. Keep the
      // shape of the base array and not the array_comp.
      auto compBaseTy = partInfo.base->getElementOrSequenceType();
      resultType = changeElementType(compBaseTy, resultType);
      assert(!partInfo.resultShape && "should not have been computed already");
      partInfo.resultShape = hlfir::genShape(loc, builder, *partInfo.base);
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

  mlir::Type visit(const Fortran::evaluate::Component &component,
                   PartInfo &partInfo) {
    if (Fortran::semantics::IsAllocatableOrPointer(component.GetLastSymbol())) {
      // In a visit, the following reference will address the target. Insert
      // the dereference here.
      partInfo.base = genWholeAllocatableOrPointerComponent(component);
      partInfo.base = hlfir::derefPointersAndAllocatables(loc, getBuilder(),
                                                          *partInfo.base);
      hlfir::genLengthParameters(loc, getBuilder(), *partInfo.base,
                                 partInfo.typeParams);
      return partInfo.base->getElementOrSequenceType();
    }
    // This function must be called from contexts where the component is not the
    // base of an ArrayRef. In these cases, the component cannot be an array
    // if the base is an array. The code below determines the shape of the
    // component reference if any.
    auto [baseType, componentType] = visitComponentImpl(component, partInfo);
    mlir::Type componentBaseType =
        hlfir::getFortranElementOrSequenceType(componentType);
    if (partInfo.base.value().isArray()) {
      // For array%scalar_comp, the result shape is
      // the one of the base. Compute it here. Note that the lower bounds of the
      // base are not the ones of the resulting reference (that are default
      // ones).
      partInfo.resultShape = hlfir::genShape(loc, getBuilder(), *partInfo.base);
      assert(!partInfo.componentShape &&
             "Fortran designators can only have one ranked part");
      return changeElementType(baseType, componentBaseType);
    }

    if (partInfo.complexPart && partInfo.componentShape) {
      // Treat ...array_comp%im/re as ...array_comp(:,:,...)%im/re
      // so that the codegen has the full slice triples for the component
      // readily available.
      fir::FirOpBuilder &builder = getBuilder();
      mlir::Type idxTy = builder.getIndexType();
      mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);

      llvm::SmallVector<mlir::Value> resultExtents;
      // Collect <lb, ub> pairs from the component shape.
      auto bounds = hlfir::genBounds(loc, builder, partInfo.componentShape);
      for (auto &boundPair : bounds) {
        // The default subscripts are <lb, ub, 1>:
        partInfo.subscripts.emplace_back(hlfir::DesignateOp::Triplet{
            boundPair.first, boundPair.second, one});
        auto extentValue = builder.genExtentFromTriplet(
            loc, boundPair.first, boundPair.second, one, idxTy);
        resultExtents.push_back(extentValue);
      }
      // The result shape is: <max((ub - lb + 1) / 1, 0), ...>.
      partInfo.resultShape = builder.genShape(loc, resultExtents);
      return componentBaseType;
    }

    // scalar%array_comp or scalar%scalar. In any case the shape of this
    // part-ref is coming from the component.
    partInfo.resultShape = partInfo.componentShape;
    partInfo.componentShape = {};
    return componentBaseType;
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
    partInfo.base = genDataRefAndSkipParentComponents(component.base());
    // If the base is an allocatable/pointer, dereference it here since the
    // component ref designates its target.
    partInfo.base =
        hlfir::derefPointersAndAllocatables(loc, builder, *partInfo.base);
    assert(partInfo.typeParams.empty() && "should not have been computed yet");

    hlfir::genLengthParameters(getLoc(), getBuilder(), *partInfo.base,
                               partInfo.typeParams);
    mlir::Type baseType = partInfo.base->getElementOrSequenceType();

    // Lower the information about the component (type, length parameters and
    // shape).
    const Fortran::semantics::Symbol &componentSym = component.GetLastSymbol();
    assert(
        !componentSym.test(Fortran::semantics::Symbol::Flag::ParentComp) &&
        "parent components are skipped and must not reach visitComponentImpl");
    partInfo.componentName = converter.getRecordTypeFieldName(componentSym);
    auto recordType =
        hlfir::getFortranElementType(baseType).cast<fir::RecordType>();
    if (recordType.isDependentType())
      TODO(getLoc(), "Designate derived type with length parameters in HLFIR");
    mlir::Type fieldType = recordType.getType(partInfo.componentName);
    assert(fieldType && "component name is not known");
    mlir::Type fieldBaseType =
        hlfir::getFortranElementOrSequenceType(fieldType);
    partInfo.componentShape = genComponentShape(componentSym, fieldBaseType);

    mlir::Type fieldEleType = hlfir::getFortranElementType(fieldBaseType);
    if (fir::isRecordWithTypeParameters(fieldEleType))
      TODO(loc,
           "lower a component that is a parameterized derived type to HLFIR");
    if (auto charTy = fieldEleType.dyn_cast<fir::CharacterType>()) {
      mlir::Location loc = getLoc();
      mlir::Type idxTy = builder.getIndexType();
      if (charTy.hasConstantLen())
        partInfo.typeParams.push_back(
            builder.createIntegerConstant(loc, idxTy, charTy.getLen()));
      else if (!Fortran::semantics::IsAllocatableOrObjectPointer(&componentSym))
        TODO(loc, "compute character length of automatic character component "
                  "in a PDT");
      // Otherwise, the length of the component is deferred and will only
      // be read when the component is dereferenced.
    }
    return {baseType, fieldType};
  }

  // Compute: "lb + (i-1)*step".
  mlir::Value computeTripletPosition(mlir::Location loc,
                                     fir::FirOpBuilder &builder,
                                     hlfir::DesignateOp::Triplet &triplet,
                                     mlir::Value oneBasedIndex) {
    mlir::Type idxTy = builder.getIndexType();
    mlir::Value lb = builder.createConvert(loc, idxTy, std::get<0>(triplet));
    mlir::Value step = builder.createConvert(loc, idxTy, std::get<2>(triplet));
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    oneBasedIndex = builder.createConvert(loc, idxTy, oneBasedIndex);
    mlir::Value zeroBased =
        builder.create<mlir::arith::SubIOp>(loc, oneBasedIndex, one);
    mlir::Value offset =
        builder.create<mlir::arith::MulIOp>(loc, zeroBased, step);
    return builder.create<mlir::arith::AddIOp>(loc, lb, offset);
  }

  /// Create an hlfir.element_addr operation to deal with vector subscripted
  /// entities. This transforms the current vector subscripted array-ref into a
  /// a scalar array-ref that is addressing the vector subscripted part given
  /// the one based indices of the hlfir.element_addr.
  /// The rest of the designator lowering will continue lowering any further
  /// parts inside the hlfir.elemental as a scalar reference.
  /// At the end of the designator lowering, the hlfir.elemental_addr will
  /// be turned into an hlfir.elemental value, unless the caller of this
  /// utility requested to get the hlfir.elemental_addr instead of lowering
  /// the designator to an mlir::Value.
  mlir::Type createVectorSubscriptElementAddrOp(
      PartInfo &partInfo, mlir::Type baseType,
      llvm::ArrayRef<mlir::Value> resultExtents) {
    fir::FirOpBuilder &builder = getBuilder();
    mlir::Value shape = builder.genShape(loc, resultExtents);
    // The type parameters to be added on the hlfir.elemental_addr are the ones
    // of the whole designator (not the ones of the vector subscripted part).
    // These are not yet known and will be added when finalizing the designator
    // lowering.
    auto elementalAddrOp =
        builder.create<hlfir::ElementalAddrOp>(loc, shape,
                                               /*isUnordered=*/true);
    setVectorSubscriptElementAddrOp(elementalAddrOp);
    builder.setInsertionPointToEnd(&elementalAddrOp.getBody().front());
    mlir::Region::BlockArgListType indices = elementalAddrOp.getIndices();
    auto indicesIterator = indices.begin();
    auto getNextOneBasedIndex = [&]() -> mlir::Value {
      assert(indicesIterator != indices.end() && "ill formed ElementalAddrOp");
      return *(indicesIterator++);
    };
    // Transform the designator into a scalar designator computing the vector
    // subscripted entity element address given one based indices (for the shape
    // of the vector subscripted designator).
    for (hlfir::DesignateOp::Subscript &subscript : partInfo.subscripts) {
      if (auto *triplet =
              std::get_if<hlfir::DesignateOp::Triplet>(&subscript)) {
        // subscript = (lb + (i-1)*step)
        mlir::Value scalarSubscript = computeTripletPosition(
            loc, builder, *triplet, getNextOneBasedIndex());
        subscript = scalarSubscript;
      } else {
        hlfir::Entity valueSubscript{std::get<mlir::Value>(subscript)};
        if (valueSubscript.isScalar())
          continue;
        // subscript = vector(i + (vector_lb-1))
        hlfir::Entity scalarSubscript = hlfir::getElementAt(
            loc, builder, valueSubscript, {getNextOneBasedIndex()});
        scalarSubscript =
            hlfir::loadTrivialScalar(loc, builder, scalarSubscript);
        subscript = scalarSubscript;
      }
    }
    builder.setInsertionPoint(elementalAddrOp);
    return baseType.cast<fir::SequenceType>().getEleTy();
  }

  /// Yield the designator for the final part-ref inside the
  /// hlfir.elemental_addr.
  void finalizeElementAddrOp(hlfir::ElementalAddrOp elementalAddrOp,
                             hlfir::EntityWithAttributes elementAddr) {
    fir::FirOpBuilder &builder = getBuilder();
    builder.setInsertionPointToEnd(&elementalAddrOp.getBody().front());
    // For polymorphic entities, it will be needed to add a mold on the
    // hlfir.elemental so that we are able to create temporary storage
    // for it using the dynamic type. It seems that a reference to the mold
    // entity can be created by evaluating the hlfir.elemental_addr
    // for a single index. The evaluation should be legal as long as
    // the hlfir.elemental_addr has no side effects, otherwise,
    // it is not clear how to get the mold reference.
    if (elementAddr.isPolymorphic())
      TODO(loc, "vector subscripted polymorphic entity in HLFIR");
    builder.create<hlfir::YieldOp>(loc, elementAddr);
    builder.setInsertionPointAfter(elementalAddrOp);
  }

  /// If the lowered designator has vector subscripts turn it into an
  /// ElementalOp, otherwise, return the lowered designator. This should
  /// only be called if the user did not request to get the
  /// hlfir.elemental_addr. In Fortran, vector subscripted designators are only
  /// writable on the left-hand side of an assignment and in input IO
  /// statements. Otherwise, they are not variables (cannot be modified, their
  /// value is taken at the place they appear).
  hlfir::EntityWithAttributes turnVectorSubscriptedDesignatorIntoValue(
      hlfir::EntityWithAttributes loweredDesignator) {
    std::optional<hlfir::ElementalAddrOp> elementalAddrOp =
        getVectorSubscriptElementAddrOp();
    if (!elementalAddrOp)
      return loweredDesignator;
    finalizeElementAddrOp(*elementalAddrOp, loweredDesignator);
    // This vector subscript designator is only being read, transform the
    // hlfir.elemental_addr into an hlfir.elemental.  The content of the
    // hlfir.elemental_addr is cloned, and the resulting address is loaded to
    // get the new element value.
    fir::FirOpBuilder &builder = getBuilder();
    mlir::Location loc = getLoc();
    mlir::Value elemental =
        hlfir::cloneToElementalOp(loc, builder, *elementalAddrOp);
    (*elementalAddrOp)->erase();
    setVectorSubscriptElementAddrOp(std::nullopt);
    fir::FirOpBuilder *bldr = &builder;
    getStmtCtx().attachCleanup(
        [=]() { bldr->create<hlfir::DestroyOp>(loc, elemental); });
    return hlfir::EntityWithAttributes{elemental};
  }

  /// Lower a subscript expression. If it is a scalar subscript that is a
  /// variable, it is loaded into an integer value. If it is an array (for
  /// vector subscripts) it is dereferenced if this is an allocatable or
  /// pointer.
  template <typename T>
  hlfir::Entity genSubscript(const Fortran::evaluate::Expr<T> &expr);

  const std::optional<hlfir::ElementalAddrOp> &
  getVectorSubscriptElementAddrOp() const {
    return vectorSubscriptElementAddrOp;
  }
  void setVectorSubscriptElementAddrOp(
      std::optional<hlfir::ElementalAddrOp> elementalAddrOp) {
    vectorSubscriptElementAddrOp = elementalAddrOp;
  }

  mlir::Location getLoc() const { return loc; }
  Fortran::lower::AbstractConverter &getConverter() { return converter; }
  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }
  Fortran::lower::SymMap &getSymMap() { return symMap; }
  Fortran::lower::StatementContext &getStmtCtx() { return stmtCtx; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  // If there is a vector subscript, an elementalAddrOp is created
  // to compute the address of the designator elements.
  std::optional<hlfir::ElementalAddrOp> vectorSubscriptElementAddrOp{};
  mlir::Location loc;
};

hlfir::EntityWithAttributes HlfirDesignatorBuilder::genDesignatorExpr(
    const Fortran::lower::SomeExpr &designatorExpr,
    bool vectorSubscriptDesignatorToValue) {
  // Expr<SomeType> plumbing to unwrap Designator<T> and call
  // gen(Designator<T>.u).
  return std::visit(
      [&](const auto &x) -> hlfir::EntityWithAttributes {
        using T = std::decay_t<decltype(x)>;
        if constexpr (Fortran::common::HasMember<
                          T, Fortran::lower::CategoryExpression>) {
          if constexpr (T::Result::category ==
                        Fortran::common::TypeCategory::Derived) {
            return gen(std::get<Fortran::evaluate::Designator<
                           Fortran::evaluate::SomeDerived>>(x.u)
                           .u,
                       vectorSubscriptDesignatorToValue);
          } else {
            return std::visit(
                [&](const auto &preciseKind) {
                  using TK =
                      typename std::decay_t<decltype(preciseKind)>::Result;
                  return gen(
                      std::get<Fortran::evaluate::Designator<TK>>(preciseKind.u)
                          .u,
                      vectorSubscriptDesignatorToValue);
                },
                x.u);
          }
        } else {
          fir::emitFatalError(loc, "unexpected typeless Designator");
        }
      },
      designatorExpr.u);
}

hlfir::ElementalAddrOp
HlfirDesignatorBuilder::convertVectorSubscriptedExprToElementalAddr(
    const Fortran::lower::SomeExpr &designatorExpr) {

  hlfir::EntityWithAttributes elementAddrEntity = genDesignatorExpr(
      designatorExpr, /*vectorSubscriptDesignatorToValue=*/false);
  assert(getVectorSubscriptElementAddrOp().has_value() &&
         "expected vector subscripts");
  hlfir::ElementalAddrOp elementalAddrOp = *getVectorSubscriptElementAddrOp();
  // Now that the type parameters have been computed, add then to the
  // hlfir.elemental_addr.
  fir::FirOpBuilder &builder = getBuilder();
  llvm::SmallVector<mlir::Value, 1> lengths;
  hlfir::genLengthParameters(loc, builder, elementAddrEntity, lengths);
  if (!lengths.empty())
    elementalAddrOp.getTypeparamsMutable().assign(lengths);
  // Create the hlfir.yield terminator inside the hlfir.elemental_body.
  builder.setInsertionPointToEnd(&elementalAddrOp.getBody().front());
  builder.create<hlfir::YieldOp>(loc, elementAddrEntity);
  builder.setInsertionPointAfter(elementalAddrOp);
  // Reset the HlfirDesignatorBuilder state, in case it is used on a new
  // designator.
  setVectorSubscriptElementAddrOp(std::nullopt);
  return elementalAddrOp;
}

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

template <int KIND>
struct BinaryOp<Fortran::evaluate::Divide<
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Complex, KIND>>> {
  using Op = Fortran::evaluate::Divide<
      Fortran::evaluate::Type<Fortran::common::TypeCategory::Complex, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs, hlfir::Entity rhs) {
    mlir::Type ty = Fortran::lower::getFIRType(
        builder.getContext(), Fortran::common::TypeCategory::Complex, KIND,
        /*params=*/std::nullopt);
    return hlfir::EntityWithAttributes{
        fir::genDivC(builder, loc, ty, lhs, rhs)};
  }
};

template <Fortran::common::TypeCategory TC, int KIND>
struct BinaryOp<Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>>> {
  using Op = Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>>;
  static hlfir::EntityWithAttributes gen(mlir::Location loc,
                                         fir::FirOpBuilder &builder, const Op &,
                                         hlfir::Entity lhs, hlfir::Entity rhs) {
    mlir::Type ty = Fortran::lower::getFIRType(builder.getContext(), TC, KIND,
                                               /*params=*/std::nullopt);
    return hlfir::EntityWithAttributes{fir::genPow(builder, loc, ty, lhs, rhs)};
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
    return hlfir::EntityWithAttributes{fir::genPow(builder, loc, ty, lhs, rhs)};
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
                                 ? fir::genMax(builder, loc, args)
                                 : fir::genMin(builder, loc, args);
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
      auto kindMap = builder.getKindMap();
      mlir::Type fromTy = lhs.getFortranElementType();
      mlir::Value origBufferSize = genCharLength(loc, builder, lhs);
      mlir::Value bufferSize{origBufferSize};
      auto fromBits = kindMap.getCharacterBitsize(
          fir::unwrapRefType(fromTy).cast<fir::CharacterType>().getFKind());
      mlir::Type toTy = Fortran::lower::getFIRType(
          builder.getContext(), TC1, KIND, /*params=*/std::nullopt);
      auto toBits = kindMap.getCharacterBitsize(
          toTy.cast<fir::CharacterType>().getFKind());
      if (toBits < fromBits) {
        // Scale by relative ratio to give a buffer of the same length.
        auto ratio = builder.createIntegerConstant(loc, bufferSize.getType(),
                                                   fromBits / toBits);
        bufferSize =
            builder.create<mlir::arith::MulIOp>(loc, bufferSize, ratio);
      }
      // allocate space on the stack for toBuffer
      auto dest = builder.create<fir::AllocaOp>(loc, toTy,
                                                mlir::ValueRange{bufferSize});
      auto src = hlfir::convertToAddress(loc, builder, lhs,
                                         lhs.getFortranElementType());
      builder.create<fir::CharConvertOp>(loc, src.first.getCharBox()->getAddr(),
                                         origBufferSize, dest);
      if (src.second.has_value())
        src.second.value()();

      return hlfir::EntityWithAttributes{builder.create<hlfir::DeclareOp>(
          loc, dest, "ctor.temp", /*shape=*/nullptr,
          /*typeparams=*/mlir::ValueRange{origBufferSize},
          fir::FortranVariableFlagsAttr{})};
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
    TODO(getLoc(), "BOZ");
  }

  hlfir::EntityWithAttributes gen(const Fortran::evaluate::NullPointer &expr) {
    auto nullop = getBuilder().create<hlfir::NullOp>(getLoc());
    return mlir::cast<fir::FortranVariableOpInterface>(nullop.getOperation());
  }

  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::ProcedureDesignator &proc) {
    return Fortran::lower::convertProcedureDesignatorToHLFIR(
        getLoc(), getConverter(), proc, getSymMap(), getStmtCtx());
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
  gen(const Fortran::evaluate::ArrayConstructor<T> &arrayCtor) {
    return Fortran::lower::ArrayConstructorBuilder<T>::gen(
        getLoc(), getConverter(), arrayCtor, getSymMap(), getStmtCtx());
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
      if (op.derived().GetType().IsUnlimitedPolymorphic())
        elementType = mlir::NoneType::get(builder.getContext());
      else
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
    mlir::Value elemental = hlfir::genElementalOp(
        loc, builder, elementType, shape, typeParams, genKernel,
        /*isUnordered=*/true, left.isPolymorphic() ? left : mlir::Value{});
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
                                                  shape, typeParams, genKernel,
                                                  /*isUnordered=*/true);
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
            .genNamedEntity(desc.base());
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
      return castResult(
          hlfir::genLBound(loc, builder, entity, desc.dimension()));
    case Fortran::evaluate::DescriptorInquiry::Field::Extent:
      return castResult(
          hlfir::genExtent(loc, builder, entity, desc.dimension()));
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
    mlir::Value value = symMap.lookupImpliedDo(toStringRef(var.name));
    if (!value)
      fir::emitFatalError(getLoc(), "ac-do-variable has no binding");
    // The index value generated by the implied-do has Index type,
    // while computations based on it inside the loop body are using
    // the original data type. So we need to cast it appropriately.
    mlir::Type varTy = getConverter().genType(toEvExpr(var));
    value = getBuilder().createConvert(getLoc(), varTy, value);
    return hlfir::EntityWithAttributes{value};
  }

  static bool
  isDerivedTypeWithLenParameters(const Fortran::semantics::Symbol &sym) {
    if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType())
      if (const Fortran::semantics::DerivedTypeSpec *derived =
              declTy->AsDerived())
        return Fortran::semantics::CountLenParameters(*derived) > 0;
    return false;
  }

  // Construct an entity holding the value specified by the
  // StructureConstructor. The initialization of the temporary entity
  // is done component by component with the help of HLFIR operations
  // ParentComponentOp, DesignateOp and AssignOp.
  hlfir::EntityWithAttributes
  gen(const Fortran::evaluate::StructureConstructor &ctor) {
    mlir::Location loc = getLoc();
    fir::FirOpBuilder &builder = getBuilder();
    mlir::Type ty = translateSomeExprToFIRType(converter, toEvExpr(ctor));
    auto recTy = ty.cast<fir::RecordType>();

    if (recTy.isDependentType())
      TODO(loc, "structure constructor for derived type with length parameters "
                "in HLFIR");

    // Allocate scalar temporary that will be initialized
    // with the values specified by the constructor.
    mlir::Value storagePtr = builder.createTemporary(loc, recTy);
    auto varOp = hlfir::EntityWithAttributes{builder.create<hlfir::DeclareOp>(
        loc, storagePtr, "ctor.temp", /*shape=*/nullptr,
        /*typeparams=*/mlir::ValueRange{}, fir::FortranVariableFlagsAttr{})};

    // Initialize any components that need initialization.
    mlir::Value box = builder.createBox(loc, fir::ExtendedValue{varOp});
    fir::runtime::genDerivedTypeInitialize(builder, loc, box);

    HlfirDesignatorBuilder designatorBuilder(loc, converter, symMap, stmtCtx);
    for (const auto &value : ctor.values()) {
      const Fortran::semantics::Symbol &sym = *value.first;
      const Fortran::lower::SomeExpr &expr = value.second.value();
      std::string name = converter.getRecordTypeFieldName(sym);
      if (sym.test(Fortran::semantics::Symbol::Flag::ParentComp)) {
        const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType();
        assert(declTypeSpec && declTypeSpec->AsDerived() &&
               "parent component symbol must have a derived type");
        mlir::Type compType = Fortran::lower::translateDerivedTypeToFIRType(
            converter, *declTypeSpec->AsDerived());
        if (fir::isRecordWithTypeParameters(compType))
          TODO(loc,
               "parent component reference with a parameterized parent type");
        mlir::Type resultType = builder.getRefType(compType);
        auto lhs = builder.create<hlfir::ParentComponentOp>(
            loc, resultType, varOp, /*shape=*/nullptr,
            /*typeparams=*/mlir::ValueRange{});
        auto rhs = gen(expr);
        builder.create<hlfir::AssignOp>(loc, rhs, lhs, /*realloc=*/false,
                                        /*keep_lhs_length_if_realloc=*/false,
                                        /*temporary_lhs=*/true);
        continue;
      }

      // Generate DesignateOp for the component.
      // The designator's result type is just a reference to the component type,
      // because the whole component is being designated.
      auto compType = recTy.getType(name);
      mlir::Value compShape =
          designatorBuilder.genComponentShape(sym, compType);
      mlir::Type designatorType = builder.getRefType(compType);

      mlir::Type fieldElemType = hlfir::getFortranElementType(compType);
      llvm::SmallVector<mlir::Value, 1> typeParams;
      if (auto charType = mlir::dyn_cast<fir::CharacterType>(fieldElemType)) {
        if (charType.hasConstantLen()) {
          mlir::Type idxType = builder.getIndexType();
          typeParams.push_back(
              builder.createIntegerConstant(loc, idxType, charType.getLen()));
        } else {
          TODO(loc, "dynamic character length in structure constructor");
        }
      }

      // Convert component symbol attributes to variable attributes.
      fir::FortranVariableFlagsAttr attrs =
          Fortran::lower::translateSymbolAttributes(builder.getContext(), sym);

      // Get the component designator.
      auto lhs = builder.create<hlfir::DesignateOp>(
          loc, designatorType, varOp, name, compShape,
          hlfir::DesignateOp::Subscripts{},
          /*substring=*/mlir::ValueRange{},
          /*complexPart=*/std::nullopt,
          /*shape=*/compShape, typeParams, attrs);

      if (attrs && bitEnumContainsAny(attrs.getFlags(),
                                      fir::FortranVariableFlagsEnum::pointer)) {
        // Pointer component construction is just a copy of the box contents.
        fir::ExtendedValue lhsExv =
            hlfir::translateToExtendedValue(loc, builder, lhs);
        auto *toBox = lhsExv.getBoxOf<fir::MutableBoxValue>();
        if (!toBox)
          fir::emitFatalError(loc, "pointer component designator could not be "
                                   "lowered to mutable box");
        Fortran::lower::associateMutableBox(converter, loc, *toBox, expr,
                                            /*lbounds=*/std::nullopt, stmtCtx);
        continue;
      }

      // Use generic assignment for all the other cases.
      bool allowRealloc =
          attrs &&
          bitEnumContainsAny(attrs.getFlags(),
                             fir::FortranVariableFlagsEnum::allocatable);
      // If the component is allocatable, then we have to check
      // whether the RHS value is allocatable or not.
      // If it is not allocatable, then AssignOp can be used directly.
      // If it is allocatable, then using AssignOp for unallocated RHS
      // will cause illegal dereference. When an unallocated allocatable
      // value is used to construct an allocatable component, the component
      // must just stay unallocated.

      // If the component is allocatable and RHS is NULL() expression, then
      // we can just skip it: the LHS must remain unallocated with its
      // defined rank.
      if (allowRealloc &&
          Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(expr))
        continue;

      // Handle special case when the initializer expression is
      // '{%SET_LENGTH(x,const_kind)}'. In structure constructor,
      // SET_LENGTH is used for initializers of character allocatable
      // components with *explicit* length, because they have to keep
      // their length regardless of the initializer expression's length.
      // We cannot just lower SET_LENGTH into hlfir.set_length in case
      // when 'x' is allocatable: if 'x' is unallocated, it is not clear
      // what hlfir.expr should be produced by hlfir.set_length.
      // So whenever the initializer expression is SET_LENGTH we
      // recognize it as the directive to keep the explicit length
      // of the LHS component, and we completely ignore 'const_kind'
      // operand assuming that it matches the LHS component's explicit
      // length. Note that in case when LHS component has deferred length,
      // the FE does not produce SET_LENGTH expression.
      //
      // When SET_LENGTH is recognized, we use 'x' as the initializer
      // for the LHS component. If 'x' is allocatable, the dynamic
      // isAllocated check will guard the assign operation as usual.
      bool keepLhsLength = false;
      hlfir::Entity rhs = std::visit(
          [&](const auto &x) -> hlfir::Entity {
            using T = std::decay_t<decltype(x)>;
            if constexpr (Fortran::common::HasMember<
                              T, Fortran::lower::CategoryExpression>) {
              if constexpr (T::Result::category ==
                            Fortran::common::TypeCategory::Character) {
                return std::visit(
                    [&](const auto &someKind) -> hlfir::Entity {
                      using T = std::decay_t<decltype(someKind)>;
                      if (const auto *setLength = std::get_if<
                              Fortran::evaluate::SetLength<T::Result::kind>>(
                              &someKind.u)) {
                        keepLhsLength = true;
                        return gen(setLength->left());
                      }

                      return gen(someKind);
                    },
                    x.u);
              }
            }
            return gen(x);
          },
          expr.u);

      if (!allowRealloc || !rhs.isMutableBox()) {
        rhs = hlfir::loadTrivialScalar(loc, builder, rhs);
        builder.create<hlfir::AssignOp>(loc, rhs, lhs, allowRealloc,
                                        allowRealloc ? keepLhsLength : false,
                                        /*temporary_lhs=*/true);
        continue;
      }

      auto [rhsExv, cleanup] =
          hlfir::translateToExtendedValue(loc, builder, rhs);
      assert(!cleanup && "unexpected cleanup");
      auto *fromBox = rhsExv.getBoxOf<fir::MutableBoxValue>();
      if (!fromBox)
        fir::emitFatalError(loc, "allocatable entity could not be lowered "
                                 "to mutable box");
      mlir::Value isAlloc =
          fir::factory::genIsAllocatedOrAssociatedTest(builder, loc, *fromBox);
      builder.genIfThen(loc, isAlloc)
          .genThen([&]() {
            rhs = hlfir::loadTrivialScalar(loc, builder, rhs);
            builder.create<hlfir::AssignOp>(loc, rhs, lhs, allowRealloc,
                                            keepLhsLength,
                                            /*temporary_lhs=*/true);
          })
          .end();
    }

    return varOp;
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
hlfir::Entity
HlfirDesignatorBuilder::genSubscript(const Fortran::evaluate::Expr<T> &expr) {
  auto loweredExpr =
      HlfirBuilder(getLoc(), getConverter(), getSymMap(), getStmtCtx())
          .gen(expr);
  fir::FirOpBuilder &builder = getBuilder();
  // Skip constant conversions that litters designators and makes generated
  // IR harder to read: directly use index constants for constant subscripts.
  mlir::Type idxTy = builder.getIndexType();
  if (!loweredExpr.isArray() && loweredExpr.getType() != idxTy)
    if (auto cstIndex = fir::getIntIfConstant(loweredExpr))
      return hlfir::EntityWithAttributes{
          builder.createIntegerConstant(getLoc(), idxTy, *cstIndex)};
  return hlfir::loadTrivialScalar(loc, builder, loweredExpr);
}

} // namespace

hlfir::EntityWithAttributes Fortran::lower::convertExprToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  return HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
}

fir::ExtendedValue Fortran::lower::convertToBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    hlfir::Entity entity, Fortran::lower::StatementContext &stmtCtx,
    mlir::Type fortranType) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto [exv, cleanup] = hlfir::convertToBox(loc, builder, entity, fortranType);
  if (cleanup)
    stmtCtx.attachCleanup(*cleanup);
  return exv;
}

fir::ExtendedValue Fortran::lower::convertExprToBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  hlfir::EntityWithAttributes loweredExpr =
      HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
  return convertToBox(loc, converter, loweredExpr, stmtCtx,
                      converter.genType(expr));
}

fir::ExtendedValue Fortran::lower::convertToAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    hlfir::Entity entity, Fortran::lower::StatementContext &stmtCtx,
    mlir::Type fortranType) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto [exv, cleanup] =
      hlfir::convertToAddress(loc, builder, entity, fortranType);
  if (cleanup)
    stmtCtx.attachCleanup(*cleanup);
  return exv;
}

fir::ExtendedValue Fortran::lower::convertExprToAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  hlfir::EntityWithAttributes loweredExpr =
      HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
  return convertToAddress(loc, converter, loweredExpr, stmtCtx,
                          converter.genType(expr));
}

fir::ExtendedValue Fortran::lower::convertToValue(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    hlfir::Entity entity, Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  auto [exv, cleanup] = hlfir::convertToValue(loc, builder, entity);
  if (cleanup)
    stmtCtx.attachCleanup(*cleanup);
  return exv;
}

fir::ExtendedValue Fortran::lower::convertExprToValue(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  hlfir::EntityWithAttributes loweredExpr =
      HlfirBuilder(loc, converter, symMap, stmtCtx).gen(expr);
  return convertToValue(loc, converter, loweredExpr, stmtCtx);
}

fir::MutableBoxValue Fortran::lower::convertExprToMutableBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap) {
  // Pointers and Allocatable cannot be temporary expressions. Temporaries may
  // be created while lowering it (e.g. if any indices expression of a
  // designator create temporaries), but they can be destroyed before using the
  // lowered pointer or allocatable;
  Fortran::lower::StatementContext localStmtCtx;
  hlfir::EntityWithAttributes loweredExpr =
      HlfirBuilder(loc, converter, symMap, localStmtCtx).gen(expr);
  fir::ExtendedValue exv = Fortran::lower::translateToExtendedValue(
      loc, converter.getFirOpBuilder(), loweredExpr, localStmtCtx);
  auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>();
  assert(mutableBox && "expression could not be lowered to mutable box");
  return *mutableBox;
}

hlfir::ElementalAddrOp
Fortran::lower::convertVectorSubscriptedExprToElementalAddr(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &designatorExpr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  return HlfirDesignatorBuilder(loc, converter, symMap, stmtCtx)
      .convertVectorSubscriptedExprToElementalAddr(designatorExpr);
}
