

//===- Interfaces.cpp - C Interface for AIIR Interfaces -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Interfaces.h"

#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Interfaces.h"
#include "aiir/CAPI/Support.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/IR/ValueRange.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/ScopeExit.h"
#include <optional>

using namespace aiir;

namespace {

std::optional<RegisteredOperationName>
getRegisteredOperationName(AiirContext context, AiirStringRef opName) {
  StringRef name(opName.data, opName.length);
  std::optional<RegisteredOperationName> info =
      RegisteredOperationName::lookup(name, unwrap(context));
  return info;
}

std::optional<Location> maybeGetLocation(AiirLocation location) {
  std::optional<Location> maybeLocation;
  if (!aiirLocationIsNull(location))
    maybeLocation = unwrap(location);
  return maybeLocation;
}

SmallVector<Value> unwrapOperands(intptr_t nOperands, AiirValue *operands) {
  SmallVector<Value> unwrappedOperands;
  (void)unwrapList(nOperands, operands, unwrappedOperands);
  return unwrappedOperands;
}

DictionaryAttr unwrapAttributes(AiirAttribute attributes) {
  DictionaryAttr attributeDict;
  if (!aiirAttributeIsNull(attributes))
    attributeDict = llvm::cast<DictionaryAttr>(unwrap(attributes));
  return attributeDict;
}

SmallVector<std::unique_ptr<Region>> unwrapRegions(intptr_t nRegions,
                                                   AiirRegion *regions) {
  // Create a vector of unique pointers to regions and make sure they are not
  // deleted when exiting the scope. This is a hack caused by C++ API expecting
  // an list of unique pointers to regions (without ownership transfer
  // semantics) and C API making ownership transfer explicit.
  SmallVector<std::unique_ptr<Region>> unwrappedRegions;
  unwrappedRegions.reserve(nRegions);
  for (intptr_t i = 0; i < nRegions; ++i)
    unwrappedRegions.emplace_back(unwrap(*(regions + i)));
  llvm::scope_exit cleaner([&]() {
    for (auto &region : unwrappedRegions)
      region.release();
  });
  return unwrappedRegions;
}

} // namespace

bool aiirOperationImplementsInterface(AiirOperation operation,
                                      AiirTypeID interfaceTypeID) {
  std::optional<RegisteredOperationName> info =
      unwrap(operation)->getRegisteredInfo();
  return info && info->hasInterface(unwrap(interfaceTypeID));
}

bool aiirOperationImplementsInterfaceStatic(AiirStringRef operationName,
                                            AiirContext context,
                                            AiirTypeID interfaceTypeID) {
  std::optional<RegisteredOperationName> info = RegisteredOperationName::lookup(
      StringRef(operationName.data, operationName.length), unwrap(context));
  return info && info->hasInterface(unwrap(interfaceTypeID));
}

AiirTypeID aiirInferTypeOpInterfaceTypeID() {
  return wrap(InferTypeOpInterface::getInterfaceID());
}

AiirLogicalResult aiirInferTypeOpInterfaceInferReturnTypes(
    AiirStringRef opName, AiirContext context, AiirLocation location,
    intptr_t nOperands, AiirValue *operands, AiirAttribute attributes,
    void *properties, intptr_t nRegions, AiirRegion *regions,
    AiirTypesCallback callback, void *userData) {
  StringRef name(opName.data, opName.length);
  std::optional<RegisteredOperationName> info =
      getRegisteredOperationName(context, opName);
  if (!info)
    return aiirLogicalResultFailure();

  std::optional<Location> maybeLocation = maybeGetLocation(location);
  SmallVector<Value> unwrappedOperands = unwrapOperands(nOperands, operands);
  DictionaryAttr attributeDict = unwrapAttributes(attributes);
  SmallVector<std::unique_ptr<Region>> unwrappedRegions =
      unwrapRegions(nRegions, regions);

  SmallVector<Type> inferredTypes;
  if (failed(info->getInterface<InferTypeOpInterface>()->inferReturnTypes(
          unwrap(context), maybeLocation, unwrappedOperands, attributeDict,
          properties, unwrappedRegions, inferredTypes)))
    return aiirLogicalResultFailure();

  SmallVector<AiirType> wrappedInferredTypes;
  wrappedInferredTypes.reserve(inferredTypes.size());
  for (Type t : inferredTypes)
    wrappedInferredTypes.push_back(wrap(t));
  callback(wrappedInferredTypes.size(), wrappedInferredTypes.data(), userData);
  return aiirLogicalResultSuccess();
}

AiirTypeID aiirInferShapedTypeOpInterfaceTypeID() {
  return wrap(InferShapedTypeOpInterface::getInterfaceID());
}

AiirLogicalResult aiirInferShapedTypeOpInterfaceInferReturnTypes(
    AiirStringRef opName, AiirContext context, AiirLocation location,
    intptr_t nOperands, AiirValue *operands, AiirAttribute attributes,
    void *properties, intptr_t nRegions, AiirRegion *regions,
    AiirShapedTypeComponentsCallback callback, void *userData) {
  std::optional<RegisteredOperationName> info =
      getRegisteredOperationName(context, opName);
  if (!info)
    return aiirLogicalResultFailure();

  std::optional<Location> maybeLocation = maybeGetLocation(location);
  SmallVector<Value> unwrappedOperands = unwrapOperands(nOperands, operands);
  DictionaryAttr attributeDict = unwrapAttributes(attributes);
  SmallVector<std::unique_ptr<Region>> unwrappedRegions =
      unwrapRegions(nRegions, regions);

  SmallVector<ShapedTypeComponents> inferredTypeComponents;
  if (failed(info->getInterface<InferShapedTypeOpInterface>()
                 ->inferReturnTypeComponents(
                     unwrap(context), maybeLocation,
                     aiir::ValueRange(llvm::ArrayRef(unwrappedOperands)),
                     attributeDict, properties, unwrappedRegions,
                     inferredTypeComponents)))
    return aiirLogicalResultFailure();

  bool hasRank;
  intptr_t rank;
  const int64_t *shapeData;
  for (const ShapedTypeComponents &t : inferredTypeComponents) {
    if (t.hasRank()) {
      hasRank = true;
      rank = t.getDims().size();
      shapeData = t.getDims().data();
    } else {
      hasRank = false;
      rank = 0;
      shapeData = nullptr;
    }
    callback(hasRank, rank, shapeData, wrap(t.getElementType()),
             wrap(t.getAttribute()), userData);
  }
  return aiirLogicalResultSuccess();
}

//===---------------------------------------------------------------------===//
// MemoryEffectOpInterface
//===---------------------------------------------------------------------===//

AiirTypeID aiirMemoryEffectsOpInterfaceTypeID() {
  return wrap(MemoryEffectOpInterface::getInterfaceID());
}

/// Fallback model for the MemoryEffectsOpInterface that uses C API callbacks.
class MemoryEffectOpInterfaceFallbackModel
    : public aiir::MemoryEffectOpInterface::FallbackModel<
          MemoryEffectOpInterfaceFallbackModel> {
public:
  /// Sets the callbacks that this FallbackModel will use.
  /// NB: the callbacks can only be set through this method as the
  /// RegisteredOperationName::attachInterface mechanism default-constructs
  /// the FallbackModel without being able to provide arguments.
  void setCallbacks(AiirMemoryEffectsOpInterfaceCallbacks callbacks) {
    this->callbacks = callbacks;
  }

  ~MemoryEffectOpInterfaceFallbackModel() {
    if (callbacks.destruct)
      callbacks.destruct(callbacks.userData);
  }

  static TypeID getInterfaceID() {
    return MemoryEffectOpInterface::getInterfaceID();
  }

  static bool classof(const aiir::MemoryEffectOpInterface::Concept *op) {
    // Enable casting back to the FallbackModel from the Interface. This is
    // necessary as attachInterface(...) default-constructs the FallbackModel
    // without being able to pass in the callbacks and returns just the Concept.
    return true;
  }

  void
  getEffects(Operation *op,
             SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    assert(callbacks.getEffects && "getEffects callback not set");
    AiirMemoryEffectInstancesList cEffects = wrap(&effects);
    callbacks.getEffects(wrap(op), cEffects, callbacks.userData);
  }

private:
  AiirMemoryEffectsOpInterfaceCallbacks callbacks;
};

/// Attach a MemoryEffectsOpInterface FallbackModel to the given named op.
/// The FallbackModel uses the provided callbacks to implement the interface.
void aiirMemoryEffectsOpInterfaceAttachFallbackModel(
    AiirContext ctx, AiirStringRef opName,
    AiirMemoryEffectsOpInterfaceCallbacks callbacks) {
  // Look up the operation definition in the context
  std::optional<RegisteredOperationName> opInfo =
      RegisteredOperationName::lookup(unwrap(opName), unwrap(ctx));

  assert(opInfo.has_value() && "operation not found in context");

  // NB: the following default-constructs the FallbackModel _without_ being able
  // to provide arguments.
  opInfo->attachInterface<MemoryEffectOpInterfaceFallbackModel>();
  // Cast to get the underlying FallbackModel and set the callbacks.
  auto *model = cast<MemoryEffectOpInterfaceFallbackModel>(
      opInfo->getInterface<MemoryEffectOpInterfaceFallbackModel>());
  assert(model && "Failed to get MemoryEffectOpInterfaceFallbackModel");
  model->setCallbacks(callbacks);
}
