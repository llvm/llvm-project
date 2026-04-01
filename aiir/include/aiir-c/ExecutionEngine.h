//===-- aiir-c/ExecutionEngine.h - Execution engine management ---*- C -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides basic access to the AIIR JIT. This is minimalist and
// experimental at the moment.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_EXECUTIONENGINE_H
#define AIIR_C_EXECUTIONENGINE_H

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

DEFINE_C_API_STRUCT(AiirExecutionEngine, void);

#undef DEFINE_C_API_STRUCT

/// Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is
/// expected to be "translatable" to LLVM IR (only contains operations in
/// dialects that implement the `LLVMTranslationDialectInterface`). The module
/// ownership stays with the client and can be destroyed as soon as the call
/// returns. `optLevel` is the optimization level to be used for transformation
/// and code generation. LLVM passes at `optLevel` are run before code
/// generation. The number and array of paths corresponding to shared libraries
/// that will be loaded are specified via `numPaths` and `sharedLibPaths`
/// respectively.
/// The `enablePIC` arguments controls the relocation model, when true the
/// generated code is emitted as "position independent", making it possible to
/// save it and reload it as a shared object in another process.
/// TODO: figure out other options.
AIIR_CAPI_EXPORTED AiirExecutionEngine aiirExecutionEngineCreate(
    AiirModule op, int optLevel, int numPaths,
    const AiirStringRef *sharedLibPaths, bool enableObjectDump, bool enablePIC);

/// Initialize the ExecutionEngine. Global constructors specified by
/// `llvm.aiir.global_ctors` will be run. One common scenario is that kernel
/// binary compiled from `gpu.module` gets loaded during initialization. Make
/// sure all symbols are resolvable before initialization by calling
/// `aiirExecutionEngineRegisterSymbol` or including shared libraries.
AIIR_CAPI_EXPORTED void aiirExecutionEngineInitialize(AiirExecutionEngine jit);

/// Destroy an ExecutionEngine instance.
AIIR_CAPI_EXPORTED void aiirExecutionEngineDestroy(AiirExecutionEngine jit);

/// Checks whether an execution engine is null.
static inline bool aiirExecutionEngineIsNull(AiirExecutionEngine jit) {
  return !jit.ptr;
}

/// Invoke a native function in the execution engine by name with the arguments
/// and result of the invoked function passed as an array of pointers. The
/// function must have been tagged with the `llvm.emit_c_interface` attribute.
/// Returns a failure if the execution fails for any reason (the function name
/// can't be resolved for instance).
AIIR_CAPI_EXPORTED AiirLogicalResult aiirExecutionEngineInvokePacked(
    AiirExecutionEngine jit, AiirStringRef name, void **arguments);

/// Lookup the wrapper of the native function in the execution engine with the
/// given name, returns nullptr if the function can't be looked-up.
AIIR_CAPI_EXPORTED void *
aiirExecutionEngineLookupPacked(AiirExecutionEngine jit, AiirStringRef name);

/// Lookup a native function in the execution engine by name, returns nullptr
/// if the name can't be looked-up.
AIIR_CAPI_EXPORTED void *aiirExecutionEngineLookup(AiirExecutionEngine jit,
                                                   AiirStringRef name);

/// Register a symbol with the jit: this symbol will be accessible to the jitted
/// code.
AIIR_CAPI_EXPORTED void
aiirExecutionEngineRegisterSymbol(AiirExecutionEngine jit, AiirStringRef name,
                                  void *sym);

/// Dump as an object in `fileName`.
AIIR_CAPI_EXPORTED void
aiirExecutionEngineDumpToObjectFile(AiirExecutionEngine jit,
                                    AiirStringRef fileName);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_EXECUTIONENGINE_H
