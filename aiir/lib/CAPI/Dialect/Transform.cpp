//===- Transform.cpp - C Interface for Transform dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Transform.h"
#include "aiir-c/Support.h"
#include "aiir/CAPI/Dialect/Transform.h"
#include "aiir/CAPI/Interfaces.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/CAPI/Rewrite.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Transform, transform,
                                      transform::TransformDialect)

//===---------------------------------------------------------------------===//
// AnyOpType
//===---------------------------------------------------------------------===//

bool aiirTypeIsATransformAnyOpType(AiirType type) {
  return isa<transform::AnyOpType>(unwrap(type));
}

AiirTypeID aiirTransformAnyOpTypeGetTypeID(void) {
  return wrap(transform::AnyOpType::getTypeID());
}

AiirType aiirTransformAnyOpTypeGet(AiirContext ctx) {
  return wrap(transform::AnyOpType::get(unwrap(ctx)));
}

AiirStringRef aiirTransformAnyOpTypeGetName(void) {
  return wrap(transform::AnyOpType::name);
}

//===---------------------------------------------------------------------===//
// AnyParamType
//===---------------------------------------------------------------------===//

bool aiirTypeIsATransformAnyParamType(AiirType type) {
  return isa<transform::AnyParamType>(unwrap(type));
}

AiirTypeID aiirTransformAnyParamTypeGetTypeID(void) {
  return wrap(transform::AnyParamType::getTypeID());
}

AiirType aiirTransformAnyParamTypeGet(AiirContext ctx) {
  return wrap(transform::AnyParamType::get(unwrap(ctx)));
}

AiirStringRef aiirTransformAnyParamTypeGetName(void) {
  return wrap(transform::AnyParamType::name);
}

//===---------------------------------------------------------------------===//
// AnyValueType
//===---------------------------------------------------------------------===//

bool aiirTypeIsATransformAnyValueType(AiirType type) {
  return isa<transform::AnyValueType>(unwrap(type));
}

AiirTypeID aiirTransformAnyValueTypeGetTypeID(void) {
  return wrap(transform::AnyValueType::getTypeID());
}

AiirType aiirTransformAnyValueTypeGet(AiirContext ctx) {
  return wrap(transform::AnyValueType::get(unwrap(ctx)));
}

AiirStringRef aiirTransformAnyValueTypeGetName(void) {
  return wrap(transform::AnyValueType::name);
}

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

bool aiirTypeIsATransformOperationType(AiirType type) {
  return isa<transform::OperationType>(unwrap(type));
}

AiirTypeID aiirTransformOperationTypeGetTypeID(void) {
  return wrap(transform::OperationType::getTypeID());
}

AiirType aiirTransformOperationTypeGet(AiirContext ctx,
                                       AiirStringRef operationName) {
  return wrap(
      transform::OperationType::get(unwrap(ctx), unwrap(operationName)));
}

AiirStringRef aiirTransformOperationTypeGetName(void) {
  return wrap(transform::OperationType::name);
}

AiirStringRef aiirTransformOperationTypeGetOperationName(AiirType type) {
  return wrap(cast<transform::OperationType>(unwrap(type)).getOperationName());
}

//===---------------------------------------------------------------------===//
// ParamType
//===---------------------------------------------------------------------===//

bool aiirTypeIsATransformParamType(AiirType type) {
  return isa<transform::ParamType>(unwrap(type));
}

AiirTypeID aiirTransformParamTypeGetTypeID(void) {
  return wrap(transform::ParamType::getTypeID());
}

AiirType aiirTransformParamTypeGet(AiirContext ctx, AiirType type) {
  return wrap(transform::ParamType::get(unwrap(ctx), unwrap(type)));
}

AiirStringRef aiirTransformParamTypeGetName(void) {
  return wrap(transform::ParamType::name);
}

AiirType aiirTransformParamTypeGetType(AiirType type) {
  return wrap(cast<transform::ParamType>(unwrap(type)).getType());
}

//===---------------------------------------------------------------------===//
// TransformRewriter
//===---------------------------------------------------------------------===//

/// Casts a `AiirTransformRewriter` to a `AiirRewriterBase`.
AiirRewriterBase aiirTransformRewriterAsBase(AiirTransformRewriter rewriter) {
  aiir::transform::TransformRewriter *t = unwrap(rewriter);
  aiir::RewriterBase *base = static_cast<aiir::RewriterBase *>(t);
  return wrap(base);
}

//===---------------------------------------------------------------------===//
// TransformResults
//===---------------------------------------------------------------------===//

void aiirTransformResultsSetOps(AiirTransformResults results, AiirValue result,
                                intptr_t numOps, AiirOperation *ops) {
  SmallVector<Operation *> opsVec;
  opsVec.reserve(numOps);
  for (intptr_t i = 0; i < numOps; ++i)
    opsVec.push_back(unwrap(ops[i]));
  unwrap(results)->set(cast<OpResult>(unwrap(result)), opsVec);
}

void aiirTransformResultsSetValues(AiirTransformResults results,
                                   AiirValue result, intptr_t numValues,
                                   AiirValue *values) {
  SmallVector<Value> valuesVec;
  valuesVec.reserve(numValues);
  for (intptr_t i = 0; i < numValues; ++i)
    valuesVec.push_back(unwrap(values[i]));
  unwrap(results)->setValues(cast<OpResult>(unwrap(result)), valuesVec);
}

void aiirTransformResultsSetParams(AiirTransformResults results,
                                   AiirValue result, intptr_t numParams,
                                   AiirAttribute *params) {
  SmallVector<Attribute> paramsVec;
  paramsVec.reserve(numParams);
  for (intptr_t i = 0; i < numParams; ++i)
    paramsVec.push_back(unwrap(params[i]));
  unwrap(results)->setParams(cast<OpResult>(unwrap(result)), paramsVec);
}

//===---------------------------------------------------------------------===//
// TransformState
//===---------------------------------------------------------------------===//

void aiirTransformStateForEachPayloadOp(AiirTransformState state,
                                        AiirValue value,
                                        AiirOperationCallback callback,
                                        void *userData) {
  for (Operation *op : unwrap(state)->getPayloadOps(unwrap(value)))
    callback(wrap(op), userData);
}

void aiirTransformStateForEachPayloadValue(AiirTransformState state,
                                           AiirValue value,
                                           AiirValueCallback callback,
                                           void *userData) {
  for (Value val : unwrap(state)->getPayloadValues(unwrap(value)))
    callback(wrap(val), userData);
}

void aiirTransformStateForEachParam(AiirTransformState state, AiirValue value,
                                    AiirAttributeCallback callback,
                                    void *userData) {
  for (Attribute attr : unwrap(state)->getParams(unwrap(value)))
    callback(wrap(attr), userData);
}

//===---------------------------------------------------------------------===//
// TransformOpInterface
//===---------------------------------------------------------------------===//

AiirTypeID aiirTransformOpInterfaceTypeID(void) {
  return wrap(transform::TransformOpInterface::getInterfaceID());
}

/// Fallback model for the TransformOpInterface that uses C API callbacks.
class TransformOpInterfaceFallbackModel
    : public aiir::transform::TransformOpInterface::FallbackModel<
          TransformOpInterfaceFallbackModel> {
public:
  /// Sets the callbacks that this FallbackModel will use.
  /// NB: the callbacks can only be set through this method as the
  /// RegisteredOperationName::attachInterface mechanism default-constructs
  /// the FallbackModel without being able to provide arguments.
  void setCallbacks(AiirTransformOpInterfaceCallbacks callbacks) {
    this->callbacks = callbacks;
  }

  ~TransformOpInterfaceFallbackModel() {
    if (callbacks.destruct)
      callbacks.destruct(callbacks.userData);
  }

  static TypeID getInterfaceID() {
    return transform::TransformOpInterface::getInterfaceID();
  }

  static bool classof(const aiir::transform::detail::
                          TransformOpInterfaceInterfaceTraits::Concept *op) {
    // Enable casting back to the FallbackModel from the Interface. This is
    // necessary as attachInterface(...) default-constructs the FallbackModel
    // without being able to pass in the callbacks and returns just the Concept.
    return true;
  }

  ::aiir::DiagnosedSilenceableFailure
  apply(Operation *op, ::aiir::transform::TransformRewriter &rewriter,
        ::aiir::transform::TransformResults &transformResults,
        ::aiir::transform::TransformState &state) const {
    assert(callbacks.apply && "apply callback not set");

    AiirDiagnosedSilenceableFailure status =
        callbacks.apply(wrap(op), wrap(&rewriter), wrap(&transformResults),
                        wrap(&state), callbacks.userData);

    switch (status) {
    case AiirDiagnosedSilenceableFailureSuccess:
      return DiagnosedSilenceableFailure::success();
    case AiirDiagnosedSilenceableFailureSilenceableFailure:
      // TODO: enable passing diagnostic info from C API to C++ API.
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(
          *(op->emitError()
            << "TransformOpInterfaceFallbackModel: silenceable failure")
               .getUnderlyingDiagnostic()));
    case AiirDiagnosedSilenceableFailureDefiniteFailure:
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    llvm_unreachable("unknown transform status");
  }

  bool allowsRepeatedHandleOperands(Operation *op) const {
    assert(callbacks.allowsRepeatedHandleOperands &&
           "allowsRepeatedHandleOperands callback not set");
    return callbacks.allowsRepeatedHandleOperands(wrap(op), callbacks.userData);
  }

private:
  AiirTransformOpInterfaceCallbacks callbacks;
};

/// Attach a TransformOpInterface FallbackModel to the given named operation.
/// The FallbackModel uses the provided callbacks to implement the interface.
void aiirTransformOpInterfaceAttachFallbackModel(
    AiirContext ctx, AiirStringRef opName,
    AiirTransformOpInterfaceCallbacks callbacks) {
  // Look up the operation definition in the context.
  std::optional<RegisteredOperationName> opInfo =
      RegisteredOperationName::lookup(unwrap(opName), unwrap(ctx));

  assert(opInfo.has_value() && "operation not found in context");

  // NB: the following default-constructs the FallbackModel _without_ being able
  // to provide arguments.
  opInfo->attachInterface<TransformOpInterfaceFallbackModel>();
  // Cast to get the underlying FallbackModel and set the callbacks.
  auto *model = cast<TransformOpInterfaceFallbackModel>(
      opInfo->getInterface<TransformOpInterfaceFallbackModel>());

  assert(model && "Failed to get TransformOpInterfaceFallbackModel");
  model->setCallbacks(callbacks);
}

//===---------------------------------------------------------------------===//
// PatternDescriptorOpInterface
//===---------------------------------------------------------------------===//

AiirTypeID aiirPatternDescriptorOpInterfaceTypeID(void) {
  return wrap(transform::PatternDescriptorOpInterface::getInterfaceID());
}

/// Fallback model for the PatternDescriptorOpInterface that uses C API
/// callbacks.
class PatternDescriptorOpInterfaceFallbackModel
    : public aiir::transform::PatternDescriptorOpInterface::FallbackModel<
          PatternDescriptorOpInterfaceFallbackModel> {
public:
  /// Sets the callbacks that this FallbackModel will use.
  /// NB: the callbacks can only be set through this method as the
  /// RegisteredOperationName::attachInterface mechanism default-constructs
  /// the FallbackModel without being able to provide arguments.
  void setCallbacks(AiirPatternDescriptorOpInterfaceCallbacks callbacks) {
    this->callbacks = callbacks;
  }

  ~PatternDescriptorOpInterfaceFallbackModel() {
    if (callbacks.destruct)
      callbacks.destruct(callbacks.userData);
  }

  static TypeID getInterfaceID() {
    return transform::PatternDescriptorOpInterface::getInterfaceID();
  }

  static bool
  classof(const aiir::transform::detail::
              PatternDescriptorOpInterfaceInterfaceTraits::Concept *op) {
    // Enable casting back to the FallbackModel from the Interface. This is
    // necessary as attachInterface(...) default-constructs the FallbackModel
    // without being able to pass in the callbacks and returns just the Concept.
    return true;
  }

  void populatePatterns(Operation *op, RewritePatternSet &patterns) const {
    assert(callbacks.populatePatterns && "populatePatterns callback not set");
    callbacks.populatePatterns(wrap(op), wrap(&patterns), callbacks.userData);
  }

  void populatePatternsWithState(Operation *op, RewritePatternSet &patterns,
                                 transform::TransformState &state) const {
    if (callbacks.populatePatternsWithState) {
      callbacks.populatePatternsWithState(wrap(op), wrap(&patterns),
                                          wrap(&state), callbacks.userData);
    } else {
      // Default implementation: call populatePatterns without state.
      populatePatterns(op, patterns);
    }
  }

private:
  AiirPatternDescriptorOpInterfaceCallbacks callbacks;
};

/// Attach a PatternDescriptorOpInterface FallbackModel to the given named
/// operation. The FallbackModel uses the provided callbacks to implement the
/// interface.
void aiirPatternDescriptorOpInterfaceAttachFallbackModel(
    AiirContext ctx, AiirStringRef opName,
    AiirPatternDescriptorOpInterfaceCallbacks callbacks) {
  // Look up the operation definition in the context.
  std::optional<RegisteredOperationName> opInfo =
      RegisteredOperationName::lookup(unwrap(opName), unwrap(ctx));

  assert(opInfo.has_value() && "operation not found in context");

  // NB: the following default-constructs the FallbackModel _without_ being able
  // to provide arguments.
  opInfo->attachInterface<PatternDescriptorOpInterfaceFallbackModel>();
  // Cast to get the underlying FallbackModel and set the callbacks.
  auto *model = cast<PatternDescriptorOpInterfaceFallbackModel>(
      opInfo->getInterface<PatternDescriptorOpInterfaceFallbackModel>());

  assert(model && "Failed to get PatternDescriptorOpInterfaceFallbackModel");
  model->setCallbacks(callbacks);
}

//===---------------------------------------------------------------------===//
// MemoryEffectsOpInterface helpers
//===---------------------------------------------------------------------===//

/// Set the effect for the operands to only read the transform handles.
void aiirTransformOnlyReadsHandle(AiirOpOperand *operands, intptr_t numOperands,
                                  AiirMemoryEffectInstancesList effects) {
  MutableArrayRef<OpOperand> operandArray(unwrap(*operands), numOperands);
  transform::onlyReadsHandle(operandArray, *unwrap(effects));
}

/// Set the effect for the operands to consuming the transform handles.
void aiirTransformConsumesHandle(AiirOpOperand *operands, intptr_t numOperands,
                                 AiirMemoryEffectInstancesList effects) {
  MutableArrayRef<OpOperand> operandArray(unwrap(*operands), numOperands);
  transform::consumesHandle(operandArray, *unwrap(effects));
}

/// Set the effect for the results to that they produce transform handles.
void aiirTransformProducesHandle(AiirValue *results, intptr_t numResults,
                                 AiirMemoryEffectInstancesList effects) {
  // NB: calling `producesHandle()` `numResults` as we cannot cast array of
  // `OpResult`s to a single `ResultRange` (and neither is `ResultRange` exposed
  // to Python). `producesHandle` iterates over the given `ResultRange` anyway.
  SmallVectorImpl<MemoryEffects::EffectInstance> &effectList = *unwrap(effects);
  for (intptr_t i = 0; i < numResults; ++i) {
    auto opResult = cast<OpResult>(unwrap(results[i]));
    transform::producesHandle(ResultRange(opResult), effectList);
  }
}

/// Set the effect of potentially modifying payload IR.
void aiirTransformModifiesPayload(AiirMemoryEffectInstancesList effects) {
  transform::modifiesPayload(*unwrap(effects));
}

/// Set the effect of potentially reading payload IR.
void aiirTransformOnlyReadsPayload(AiirMemoryEffectInstancesList effects) {
  transform::onlyReadsPayload(*unwrap(effects));
}
