//===-- aiir-c/Interfaces.h - C API to Core AIIR IR interfaces ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to AIIR interface classes. It is
// intended to contain interfaces defined in lib/Interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_INTERFACES_H
#define AIIR_C_INTERFACES_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AiirMemoryEffectInstancesList, void);

#undef DEFINE_C_API_STRUCT

/// Returns `true` if the given operation implements an interface identified by
/// its TypeID.
AIIR_CAPI_EXPORTED bool
aiirOperationImplementsInterface(AiirOperation operation,
                                 AiirTypeID interfaceTypeID);

/// Returns `true` if the operation identified by its canonical string name
/// implements the interface identified by its TypeID in the given context.
/// Note that interfaces may be attached to operations in some contexts and not
/// others.
AIIR_CAPI_EXPORTED bool
aiirOperationImplementsInterfaceStatic(AiirStringRef operationName,
                                       AiirContext context,
                                       AiirTypeID interfaceTypeID);

//===----------------------------------------------------------------------===//
// InferTypeOpInterface.
//===----------------------------------------------------------------------===//

/// Returns the interface TypeID of the InferTypeOpInterface.
AIIR_CAPI_EXPORTED AiirTypeID aiirInferTypeOpInterfaceTypeID(void);

/// These callbacks are used to return multiple types from functions while
/// transferring ownership to the caller. The first argument is the number of
/// consecutive elements pointed to by the second argument. The third argument
/// is an opaque pointer forwarded to the callback by the caller.
typedef void (*AiirTypesCallback)(intptr_t, AiirType *, void *);

/// Infers the return types of the operation identified by its canonical given
/// the arguments that will be supplied to its generic builder. Calls `callback`
/// with the types of inferred arguments, potentially several times, on success.
/// Returns failure otherwise.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirInferTypeOpInterfaceInferReturnTypes(
    AiirStringRef opName, AiirContext context, AiirLocation location,
    intptr_t nOperands, AiirValue *operands, AiirAttribute attributes,
    void *properties, intptr_t nRegions, AiirRegion *regions,
    AiirTypesCallback callback, void *userData);

//===----------------------------------------------------------------------===//
// InferShapedTypeOpInterface.
//===----------------------------------------------------------------------===//

/// Returns the interface TypeID of the InferShapedTypeOpInterface.
AIIR_CAPI_EXPORTED AiirTypeID aiirInferShapedTypeOpInterfaceTypeID(void);

/// These callbacks are used to return multiple shaped type components from
/// functions while transferring ownership to the caller. The first argument is
/// the has rank boolean followed by the the rank and a pointer to the shape
/// (if applicable). The next argument is the element type, then the attribute.
/// The last argument is an opaque pointer forwarded to the callback by the
/// caller. This callback will be called potentially multiple times for each
/// shaped type components.
typedef void (*AiirShapedTypeComponentsCallback)(bool, intptr_t,
                                                 const int64_t *, AiirType,
                                                 AiirAttribute, void *);

/// Infers the return shaped type components of the operation. Calls `callback`
/// with the types of inferred arguments on success. Returns failure otherwise.
AIIR_CAPI_EXPORTED AiirLogicalResult
aiirInferShapedTypeOpInterfaceInferReturnTypes(
    AiirStringRef opName, AiirContext context, AiirLocation location,
    intptr_t nOperands, AiirValue *operands, AiirAttribute attributes,
    void *properties, intptr_t nRegions, AiirRegion *regions,
    AiirShapedTypeComponentsCallback callback, void *userData);

//===---------------------------------------------------------------------===//
// MemoryEffectsOpInterface
//===---------------------------------------------------------------------===//

/// Returns the interface TypeID of the MemoryEffectsOpInterface.
AIIR_CAPI_EXPORTED AiirTypeID aiirMemoryEffectsOpInterfaceTypeID(void);

/// Callbacks for implementing MemoryEffectsOpInterface from external code.
typedef struct {
  /// Optional constructor for user data. Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for user data. Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// Get memory effects callback.
  void (*getEffects)(AiirOperation op, AiirMemoryEffectInstancesList effects,
                     void *userData);
  void *userData;
} AiirMemoryEffectsOpInterfaceCallbacks;

/// Attach a new FallbackModel for the MemoryEffectsOpInterface to the named
/// operation. The FallbackModel will call the provided callbacks.
AIIR_CAPI_EXPORTED void aiirMemoryEffectsOpInterfaceAttachFallbackModel(
    AiirContext ctx, AiirStringRef opName,
    AiirMemoryEffectsOpInterfaceCallbacks callbacks);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_INTERFACES_H
