//===-- aiir-c/Dialect/Transform/Interpreter.h --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C interface to the transform dialect interpreter.
//
//===----------------------------------------------------------------------===//

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

DEFINE_C_API_STRUCT(AiirTransformOptions, void);

#undef DEFINE_C_API_STRUCT

//----------------------------------------------------------------------------//
// AiirTransformOptions
//----------------------------------------------------------------------------//

/// Creates a default-initialized transform options object.
AIIR_CAPI_EXPORTED AiirTransformOptions aiirTransformOptionsCreate(void);

/// Enables or disables expensive checks in transform options.
AIIR_CAPI_EXPORTED void
aiirTransformOptionsEnableExpensiveChecks(AiirTransformOptions transformOptions,
                                          bool enable);

/// Returns true if expensive checks are enabled in transform options.
AIIR_CAPI_EXPORTED bool aiirTransformOptionsGetExpensiveChecksEnabled(
    AiirTransformOptions transformOptions);

/// Enables or disables the enforcement of the top-level transform op being
/// single in transform options.
AIIR_CAPI_EXPORTED void aiirTransformOptionsEnforceSingleTopLevelTransformOp(
    AiirTransformOptions transformOptions, bool enable);

/// Returns true if the enforcement of the top-level transform op being single
/// is enabled in transform options.
AIIR_CAPI_EXPORTED bool aiirTransformOptionsGetEnforceSingleTopLevelTransformOp(
    AiirTransformOptions transformOptions);

/// Destroys a transform options object previously created by
/// aiirTransformOptionsCreate.
AIIR_CAPI_EXPORTED void
aiirTransformOptionsDestroy(AiirTransformOptions transformOptions);

//----------------------------------------------------------------------------//
// Transform interpreter and utilities.
//----------------------------------------------------------------------------//

/// Applies the transformation script starting at the given transform root
/// operation to the given payload operation. The module containing the
/// transform root as well as the transform options should be provided. The
/// transform operation must implement TransformOpInterface and the module must
/// be a ModuleOp. Returns the status of the application.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirTransformApplyNamedSequence(
    AiirOperation payload, AiirOperation transformRoot,
    AiirOperation transformModule, AiirTransformOptions transformOptions);

/// Merge the symbols from `other` into `target`, potentially renaming them to
/// avoid conflicts. Private symbols may be renamed during the merge, public
/// symbols must have at most one declaration. A name conflict in public symbols
/// is reported as an error before returning a failure.
///
/// Note that this clones the `other` operation unlike the C++ counterpart that
/// takes ownership.
AIIR_CAPI_EXPORTED AiirLogicalResult
aiirMergeSymbolsIntoFromClone(AiirOperation target, AiirOperation other);

#ifdef __cplusplus
}
#endif
