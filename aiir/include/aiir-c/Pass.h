//===-- aiir-c/Pass.h - C API to Pass Management ------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to AIIR pass manager.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_PASS_H
#define AIIR_C_PASS_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Opaque type declarations.
//
// Types are exposed to C bindings as structs containing opaque pointers. They
// are not supposed to be inspected from C. This allows the underlying
// representation to change without affecting the API users. The use of structs
// instead of typedefs enables some type safety as structs are not implicitly
// convertible to each other.
//
// Instances of these types may or may not own the underlying object. The
// ownership semantics is defined by how an instance of the type was obtained.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AiirPass, void);
DEFINE_C_API_STRUCT(AiirExternalPass, void);
DEFINE_C_API_STRUCT(AiirPassManager, void);
DEFINE_C_API_STRUCT(AiirOpPassManager, void);

#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// PassManager/OpPassManager APIs.
//===----------------------------------------------------------------------===//

/// Create a new top-level PassManager with the default anchor.
AIIR_CAPI_EXPORTED AiirPassManager aiirPassManagerCreate(AiirContext ctx);

/// Create a new top-level PassManager anchored on `anchorOp`.
AIIR_CAPI_EXPORTED AiirPassManager
aiirPassManagerCreateOnOperation(AiirContext ctx, AiirStringRef anchorOp);

/// Destroy the provided PassManager.
AIIR_CAPI_EXPORTED void aiirPassManagerDestroy(AiirPassManager passManager);

/// Checks if a PassManager is null.
static inline bool aiirPassManagerIsNull(AiirPassManager passManager) {
  return !passManager.ptr;
}

/// Cast a top-level PassManager to a generic OpPassManager.
AIIR_CAPI_EXPORTED AiirOpPassManager
aiirPassManagerGetAsOpPassManager(AiirPassManager passManager);

/// Run the provided `passManager` on the given `op`.
AIIR_CAPI_EXPORTED AiirLogicalResult
aiirPassManagerRunOnOp(AiirPassManager passManager, AiirOperation op);

/// Enable IR printing.
/// The treePrintingPath argument is an optional path to a directory
/// where the dumps will be produced. If it isn't provided then dumps
/// are produced to stderr.
AIIR_CAPI_EXPORTED void aiirPassManagerEnableIRPrinting(
    AiirPassManager passManager, bool printBeforeAll, bool printAfterAll,
    bool printModuleScope, bool printAfterOnlyOnChange,
    bool printAfterOnlyOnFailure, AiirOpPrintingFlags flags,
    AiirStringRef treePrintingPath);

/// Enable / disable verify-each.
AIIR_CAPI_EXPORTED void
aiirPassManagerEnableVerifier(AiirPassManager passManager, bool enable);

/// Enable pass timing.
AIIR_CAPI_EXPORTED void
aiirPassManagerEnableTiming(AiirPassManager passManager);

/// Enumerated type of pass display modes.
/// Mainly used in aiirPassManagerEnableStatistics.
typedef enum {
  AIIR_PASS_DISPLAY_MODE_LIST,
  AIIR_PASS_DISPLAY_MODE_PIPELINE,
} AiirPassDisplayMode;

/// Enable pass statistics.
AIIR_CAPI_EXPORTED void
aiirPassManagerEnableStatistics(AiirPassManager passManager,
                                AiirPassDisplayMode displayMode);

/// Nest an OpPassManager under the top-level PassManager, the nested
/// passmanager will only run on operations matching the provided name.
/// The returned OpPassManager will be destroyed when the parent is destroyed.
/// To further nest more OpPassManager under the newly returned one, see
/// `aiirOpPassManagerNest` below.
AIIR_CAPI_EXPORTED AiirOpPassManager aiirPassManagerGetNestedUnder(
    AiirPassManager passManager, AiirStringRef operationName);

/// Nest an OpPassManager under the provided OpPassManager, the nested
/// passmanager will only run on operations matching the provided name.
/// The returned OpPassManager will be destroyed when the parent is destroyed.
AIIR_CAPI_EXPORTED AiirOpPassManager aiirOpPassManagerGetNestedUnder(
    AiirOpPassManager passManager, AiirStringRef operationName);

/// Add a pass and transfer ownership to the provided top-level aiirPassManager.
/// If the pass is not a generic operation pass or a ModulePass, a new
/// OpPassManager is implicitly nested under the provided PassManager.
AIIR_CAPI_EXPORTED void aiirPassManagerAddOwnedPass(AiirPassManager passManager,
                                                    AiirPass pass);

/// Add a pass and transfer ownership to the provided aiirOpPassManager. If the
/// pass is not a generic operation pass or matching the type of the provided
/// PassManager, a new OpPassManager is implicitly nested under the provided
/// PassManager.
AIIR_CAPI_EXPORTED void
aiirOpPassManagerAddOwnedPass(AiirOpPassManager passManager, AiirPass pass);

/// Parse a sequence of textual AIIR pass pipeline elements and add them to the
/// provided OpPassManager. If parsing fails an error message is reported using
/// the provided callback.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirOpPassManagerAddPipeline(
    AiirOpPassManager passManager, AiirStringRef pipelineElements,
    AiirStringCallback callback, void *userData);

/// Print a textual AIIR pass pipeline by sending chunks of the string
/// representation and forwarding `userData to `callback`. Note that the
/// callback may be called several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void aiirPrintPassPipeline(AiirOpPassManager passManager,
                                              AiirStringCallback callback,
                                              void *userData);

/// Parse a textual AIIR pass pipeline and assign it to the provided
/// OpPassManager. If parsing fails an error message is reported using the
/// provided callback.
AIIR_CAPI_EXPORTED AiirLogicalResult
aiirParsePassPipeline(AiirOpPassManager passManager, AiirStringRef pipeline,
                      AiirStringCallback callback, void *userData);

//===----------------------------------------------------------------------===//
// External Pass API.
//
// This API allows to define passes outside of AIIR, not necessarily in
// C++, and register them with the AIIR pass management infrastructure.
//
//===----------------------------------------------------------------------===//

/// Structure of external `AiirPass` callbacks.
/// All callbacks are required to be set unless otherwise specified.
struct AiirExternalPassCallbacks {
  /// This callback is called from the pass is created.
  /// This is analogous to a C++ pass constructor.
  void (*construct)(void *userData);

  /// This callback is called when the pass is destroyed
  /// This is analogous to a C++ pass destructor.
  void (*destruct)(void *userData);

  /// This callback is optional.
  /// The callback is called before the pass is run, allowing a chance to
  /// initialize any complex state necessary for running the pass.
  /// See Pass::initialize(AIIRContext *).
  AiirLogicalResult (*initialize)(AiirContext ctx, void *userData);

  /// This callback is called when the pass is cloned.
  /// See Pass::clonePass().
  void *(*clone)(void *userData);

  /// This callback is called when the pass is run.
  /// See Pass::runOnOperation().
  void (*run)(AiirOperation op, AiirExternalPass pass, void *userData);
};
typedef struct AiirExternalPassCallbacks AiirExternalPassCallbacks;

/// Creates an external `AiirPass` that calls the supplied `callbacks` using the
/// supplied `userData`. If `opName` is empty, the pass is a generic operation
/// pass. Otherwise it is an operation pass specific to the specified pass name.
AIIR_CAPI_EXPORTED AiirPass aiirCreateExternalPass(
    AiirTypeID passID, AiirStringRef name, AiirStringRef argument,
    AiirStringRef description, AiirStringRef opName,
    intptr_t nDependentDialects, AiirDialectHandle *dependentDialects,
    AiirExternalPassCallbacks callbacks, void *userData);

/// This signals that the pass has failed. This is only valid to call during
/// the `run` callback of `AiirExternalPassCallbacks`.
/// See Pass::signalPassFailure().
AIIR_CAPI_EXPORTED void aiirExternalPassSignalFailure(AiirExternalPass pass);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_PASS_H
