//===-- aiir-c/Dialect/Transform.h - C API for Transform Dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_TRANSFORM_H
#define AIIR_C_DIALECT_TRANSFORM_H

#include "aiir-c/IR.h"
#include "aiir-c/Interfaces.h"
#include "aiir-c/Rewrite.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Transform, transform);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AiirTransformResults, void);
DEFINE_C_API_STRUCT(AiirTransformRewriter, void);
DEFINE_C_API_STRUCT(AiirTransformState, void);

#undef DEFINE_C_API_STRUCT

//===---------------------------------------------------------------------===//
// DiagnosedSilenceableFailure
//===---------------------------------------------------------------------===//

/// Enum representing the result of a transform operation.
typedef enum {
  /// The operation succeeded.
  AiirDiagnosedSilenceableFailureSuccess,
  /// The operation failed in a silenceable way.
  AiirDiagnosedSilenceableFailureSilenceableFailure,
  /// The operation failed definitively.
  AiirDiagnosedSilenceableFailureDefiniteFailure
} AiirDiagnosedSilenceableFailure;

//===---------------------------------------------------------------------===//
// AnyOpType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsATransformAnyOpType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirTransformAnyOpTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirTransformAnyOpTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirTransformAnyOpTypeGetName(void);

//===---------------------------------------------------------------------===//
// AnyParamType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsATransformAnyParamType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirTransformAnyParamTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirTransformAnyParamTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirTransformAnyParamTypeGetName(void);

//===---------------------------------------------------------------------===//
// AnyValueType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsATransformAnyValueType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirTransformAnyValueTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirTransformAnyValueTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirTransformAnyValueTypeGetName(void);

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsATransformOperationType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirTransformOperationTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType
aiirTransformOperationTypeGet(AiirContext ctx, AiirStringRef operationName);

AIIR_CAPI_EXPORTED AiirStringRef aiirTransformOperationTypeGetName(void);

AIIR_CAPI_EXPORTED AiirStringRef
aiirTransformOperationTypeGetOperationName(AiirType type);

//===---------------------------------------------------------------------===//
// ParamType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsATransformParamType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirTransformParamTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirTransformParamTypeGet(AiirContext ctx,
                                                      AiirType type);

AIIR_CAPI_EXPORTED AiirStringRef aiirTransformParamTypeGetName(void);

AIIR_CAPI_EXPORTED AiirType aiirTransformParamTypeGetType(AiirType type);

//===---------------------------------------------------------------------===//
// TransformRewriter
//===---------------------------------------------------------------------===//

/// Cast the TransformRewriter to a RewriterBase
AIIR_CAPI_EXPORTED AiirRewriterBase
aiirTransformRewriterAsBase(AiirTransformRewriter rewriter);

//===---------------------------------------------------------------------===//
// TransformResults
//===---------------------------------------------------------------------===//

/// Set the payload operations for a transform result by iterating over a list.
AIIR_CAPI_EXPORTED void aiirTransformResultsSetOps(AiirTransformResults results,
                                                   AiirValue result,
                                                   intptr_t numOps,
                                                   AiirOperation *ops);

/// Set the payload values for a transform result by iterating over a list.
AIIR_CAPI_EXPORTED void
aiirTransformResultsSetValues(AiirTransformResults results, AiirValue result,
                              intptr_t numValues, AiirValue *values);

/// Set the parameters for a transform result by iterating over a list.
AIIR_CAPI_EXPORTED void
aiirTransformResultsSetParams(AiirTransformResults results, AiirValue result,
                              intptr_t numParams, AiirAttribute *params);

//===---------------------------------------------------------------------===//
// TransformState
//===---------------------------------------------------------------------===//

/// Callback for iterating over payload operations.
typedef void (*AiirOperationCallback)(AiirOperation, void *userData);

/// Iterate over payload operations associated with the transform IR value.
/// Calls the callback for each payload operation.
AIIR_CAPI_EXPORTED void
aiirTransformStateForEachPayloadOp(AiirTransformState state, AiirValue value,
                                   AiirOperationCallback callback,
                                   void *userData);

/// Callback for iterating over payload values.
typedef void (*AiirValueCallback)(AiirValue, void *userData);

/// Iterate over payload values associated with the transform IR value.
/// Calls the callback for each payload value.
AIIR_CAPI_EXPORTED void
aiirTransformStateForEachPayloadValue(AiirTransformState state, AiirValue value,
                                      AiirValueCallback callback,
                                      void *userData);

/// Callback for iterating over parameters.
typedef void (*AiirAttributeCallback)(AiirAttribute, void *userData);

/// Iterate over parameters associated with the transform IR value.
/// Calls the callback for each parameter.
AIIR_CAPI_EXPORTED void
aiirTransformStateForEachParam(AiirTransformState state, AiirValue value,
                               AiirAttributeCallback callback, void *userData);

//===---------------------------------------------------------------------===//
// TransformOpInterface
//===---------------------------------------------------------------------===//

/// Returns the interface TypeID of the TransformOpInterface.
AIIR_CAPI_EXPORTED AiirTypeID aiirTransformOpInterfaceTypeID(void);

/// Callbacks for implementing TransformOpInterface from external code.
typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// Apply callback that implements the transformation.
  AiirDiagnosedSilenceableFailure (*apply)(AiirOperation op,
                                           AiirTransformRewriter rewriter,
                                           AiirTransformResults results,
                                           AiirTransformState state,
                                           void *userData);
  /// Callback to check if repeated handle operands are allowed.
  bool (*allowsRepeatedHandleOperands)(AiirOperation op, void *userData);
  void *userData;
} AiirTransformOpInterfaceCallbacks;

/// Attach TransformOpInterface to the operation with the given name using
/// the provided callbacks.
AIIR_CAPI_EXPORTED void aiirTransformOpInterfaceAttachFallbackModel(
    AiirContext ctx, AiirStringRef opName,
    AiirTransformOpInterfaceCallbacks callbacks);

//===---------------------------------------------------------------------===//
// PatternDescriptorOpInterface
//===---------------------------------------------------------------------===//

/// Returns the interface TypeID of the PatternDescriptorOpInterface.
AIIR_CAPI_EXPORTED AiirTypeID aiirPatternDescriptorOpInterfaceTypeID(void);

/// Callbacks for implementing PatternDescriptorOpInterface from external code.
typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// Callback to populate rewrite patterns into the given pattern set.
  void (*populatePatterns)(AiirOperation op, AiirRewritePatternSet patterns,
                           void *userData);
  /// Optional callback to populate rewrite patterns with transform state.
  /// Set to nullptr to use the default implementation (calls populatePatterns).
  void (*populatePatternsWithState)(AiirOperation op,
                                    AiirRewritePatternSet patterns,
                                    AiirTransformState state, void *userData);
  void *userData;
} AiirPatternDescriptorOpInterfaceCallbacks;

/// Attach PatternDescriptorOpInterface to the operation with the given name
/// using the provided callbacks.
AIIR_CAPI_EXPORTED void aiirPatternDescriptorOpInterfaceAttachFallbackModel(
    AiirContext ctx, AiirStringRef opName,
    AiirPatternDescriptorOpInterfaceCallbacks callbacks);

//===---------------------------------------------------------------------===//
// Transform-specifc MemoryEffectsOpInterface helpers
//===---------------------------------------------------------------------===//

/// Helper to mark operands as only reading handles.
AIIR_CAPI_EXPORTED void
aiirTransformOnlyReadsHandle(AiirOpOperand *operands, intptr_t numOperands,
                             AiirMemoryEffectInstancesList effects);

/// Helper to mark operands as consuming handles.
AIIR_CAPI_EXPORTED void
aiirTransformConsumesHandle(AiirOpOperand *operands, intptr_t numOperands,
                            AiirMemoryEffectInstancesList effects);

/// Helper to mark results as producing handles.
AIIR_CAPI_EXPORTED void
aiirTransformProducesHandle(AiirValue *results, intptr_t numResults,
                            AiirMemoryEffectInstancesList effects);

/// Helper to mark potential modifications to the payload IR.
AIIR_CAPI_EXPORTED void
aiirTransformModifiesPayload(AiirMemoryEffectInstancesList effects);

/// Helper to mark potential reads from the payload IR.
AIIR_CAPI_EXPORTED void
aiirTransformOnlyReadsPayload(AiirMemoryEffectInstancesList effects);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/Transform/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_TRANSFORM_H
