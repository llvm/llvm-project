/*==-- clang-c/CAS.h - CAS Interface ------------------------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides interfaces for creating and working with CAS and      *|
|* ActionCache interfaces.                                                    *|
|*                                                                            *|
|* An example of its usage is available in c-index-test/core_main.cpp.        *|
|*                                                                            *|
|* EXPERIMENTAL: These interfaces are experimental and will change. If you    *|
|* use these be prepared for them to change without notice on any commit.     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_CLANG_C_CAS_H
#define LLVM_CLANG_C_CAS_H

#include "clang-c/CXErrorCode.h"
#include "clang-c/CXString.h"
#include "clang-c/Platform.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup CAS CAS and ActionCache interface.
 * @{
 */

/**
 * Configuration options for ObjectStore and ActionCache.
 */
typedef struct CXOpaqueCASOptions *CXCASOptions;

/**
 * Encapsulates instances of ObjectStore and ActionCache, created from a
 * particular configuration of \p CXCASOptions.
 */
typedef struct CXOpaqueCASDatabases *CXCASDatabases;

/**
 * Content-addressable storage for objects.
 */
typedef struct CXOpaqueCASObjectStore *CXCASObjectStore;

/**
 * A cache from a key describing an action to the result of doing it.
 */
typedef struct CXOpaqueCASActionCache *CXCASActionCache;

typedef struct CXOpaqueCASObject *CXCASObject;

/**
 * Result of \c clang_experimental_cas_getCachedCompilation.
 */
typedef struct CXOpaqueCASCachedCompilation *CXCASCachedCompilation;

/**
 * Result of \c clang_experimental_cas_replayCompilation.
 */
typedef struct CXOpaqueCASReplayResult *CXCASReplayResult;

/**
 * Used for cancelling asynchronous actions.
 */
typedef struct CXOpaqueCASCancellationToken *CXCASCancellationToken;

/**
 * Create a \c CXCASOptions object.
 */
CINDEX_LINKAGE CXCASOptions clang_experimental_cas_Options_create(void);

/**
 * Dispose of a \c CXCASOptions object.
 */
CINDEX_LINKAGE void clang_experimental_cas_Options_dispose(CXCASOptions);

/**
 * Configure the file path to use for on-disk CAS/cache instances.
 */
CINDEX_LINKAGE void
clang_experimental_cas_Options_setOnDiskPath(CXCASOptions, const char *Path);

/**
 * Configure the path to a library that implements the LLVM CAS plugin API.
 */
CINDEX_LINKAGE void
clang_experimental_cas_Options_setPluginPath(CXCASOptions, const char *Path);

/**
 * Set a value for a named option that the CAS plugin supports.
 */
CINDEX_LINKAGE void
clang_experimental_cas_Options_setPluginOption(CXCASOptions, const char *Name,
                                               const char *Value);

/**
 * Creates instances for a CAS object store and action cache based on the
 * configuration of a \p CXCASOptions.
 *
 * \param Opts configuration options.
 * \param[out] Error The error string to pass back to client (if any).
 *
 * \returns The resulting instances object, or null if there was an error.
 */
CINDEX_LINKAGE CXCASDatabases
clang_experimental_cas_Databases_create(CXCASOptions Opts, CXString *Error);

/**
 * Dispose of a \c CXCASDatabases object.
 */
CINDEX_LINKAGE void clang_experimental_cas_Databases_dispose(CXCASDatabases);

/**
 * Get the local storage size of the CAS/cache data in bytes.
 *
 * \param[out] OutError The error object to pass back to client (if any).
 * If non-null the object must be disposed using \c clang_Error_dispose.
 * \returns the local storage size of the CAS/cache data, or -1 if the
 * implementation does not support reporting such size, or -2 if an error
 * occurred.
 */
CINDEX_LINKAGE int64_t clang_experimental_cas_Databases_get_storage_size(
    CXCASDatabases, CXError *OutError);

/**
 * Set the size for limiting disk storage growth.
 *
 * \param size_limit the maximum size limit in bytes. 0 means no limit. Negative
 * values are invalid.
 * \returns an error object if there was an error, NULL otherwise.
 * If non-null the object must be disposed using \c clang_Error_dispose.
 */
CINDEX_LINKAGE CXError clang_experimental_cas_Databases_set_size_limit(
    CXCASDatabases, int64_t size_limit);

/**
 * Prune local storage to reduce its size according to the desired size limit.
 * Pruning can happen concurrently with other operations.
 *
 * \returns an error object if there was an error, NULL otherwise.
 * If non-null the object must be disposed using \c clang_Error_dispose.
 */
CINDEX_LINKAGE
CXError clang_experimental_cas_Databases_prune_ondisk_data(CXCASDatabases);

/**
 * Loads an object using its printed \p CASID.
 *
 * \param CASID The printed CASID string for the object.
 * \param[out] OutError The error object to pass back to client (if any).
 * If non-null the object must be disposed using \c clang_Error_dispose.
 *
 * \returns The resulting object, or null if the object was not found or an
 * error occurred. The object should be disposed using
 * \c clang_experimental_cas_CASObject_dispose.
 */
CINDEX_LINKAGE CXCASObject clang_experimental_cas_loadObjectByString(
    CXCASDatabases, const char *CASID, CXError *OutError);

/**
 * Asynchronous version of \c clang_experimental_cas_loadObjectByString.
 *
 * \param CASID The printed CASID string for the object.
 * \param Ctx opaque value to pass to the callback.
 * \param Callback receives a \c CXCASObject, or \c CXError if an error occurred
 * or both NULL if the object was not found or the call was cancelled.
 * The objects should be disposed with
 * \c clang_experimental_cas_CASObject_dispose or \c clang_Error_dispose.
 * \param[out] OutToken if non-null receives a \c CXCASCancellationToken that
 * can be used to cancel the call using
 * \c clang_experimental_cas_CancellationToken_cancel. The object should be
 * disposed using \c clang_experimental_cas_CancellationToken_dispose.
 */
CINDEX_LINKAGE void clang_experimental_cas_loadObjectByString_async(
    CXCASDatabases, const char *CASID, void *Ctx,
    void (*Callback)(void *Ctx, CXCASObject, CXError),
    CXCASCancellationToken *OutToken);

/**
 * Dispose of a \c CXCASObject object.
 */
CINDEX_LINKAGE void clang_experimental_cas_CASObject_dispose(CXCASObject);

/**
 * Looks up a cache key and returns the associated set of compilation output IDs
 *
 * \param CacheKey The printed compilation cache key string.
 * \param Globally if true it is a hint to the underlying CAS implementation
 * that the lookup is profitable to be done on a distributed caching level, not
 * just locally.
 * \param[out] OutError The error object to pass back to client (if any).
 * If non-null the object must be disposed using \c clang_Error_dispose.
 *
 * \returns The resulting object, or null if the cache key was not found or an
 * error occurred. The object should be disposed using
 * \c clang_experimental_cas_CachedCompilation_dispose.
 */
CINDEX_LINKAGE CXCASCachedCompilation
clang_experimental_cas_getCachedCompilation(CXCASDatabases,
                                            const char *CacheKey, bool Globally,
                                            CXError *OutError);

/**
 * Asynchronous version of \c clang_experimental_cas_getCachedCompilation.
 *
 * \param CacheKey The printed compilation cache key string.
 * \param Globally if true it is a hint to the underlying CAS implementation
 * that the lookup is profitable to be done on a distributed caching level, not
 * just locally.
 * \param Ctx opaque value to pass to the callback.
 * \param Callback receives a \c CXCASCachedCompilation, or \c CXError if an
 * error occurred or both NULL if the object was not found or the call was
 * cancelled. The objects should be disposed with
 * \c clang_experimental_cas_CachedCompilation_dispose or \c clang_Error_dispose
 * \param[out] OutToken if non-null receives a \c CXCASCancellationToken that
 * can be used to cancel the call using
 * \c clang_experimental_cas_CancellationToken_cancel. The object should be
 * disposed using \c clang_experimental_cas_CancellationToken_dispose.
 */
CINDEX_LINKAGE void clang_experimental_cas_getCachedCompilation_async(
    CXCASDatabases, const char *CacheKey, bool Globally, void *Ctx,
    void (*Callback)(void *Ctx, CXCASCachedCompilation, CXError),
    CXCASCancellationToken *OutToken);

/**
 * Dispose of a \c CXCASCachedCompilation object.
 */
CINDEX_LINKAGE void
    clang_experimental_cas_CachedCompilation_dispose(CXCASCachedCompilation);

/**
 * \returns number of compilation outputs.
 */
CINDEX_LINKAGE size_t clang_experimental_cas_CachedCompilation_getNumOutputs(
    CXCASCachedCompilation);

/**
 * \returns the compilation output name given the index via \p OutputIdx.
 */
CINDEX_LINKAGE CXString clang_experimental_cas_CachedCompilation_getOutputName(
    CXCASCachedCompilation, size_t OutputIdx);

/**
 * \returns the compilation output printed CASID given the index via
 * \p OutputIdx.
 */
CINDEX_LINKAGE CXString
clang_experimental_cas_CachedCompilation_getOutputCASIDString(
    CXCASCachedCompilation, size_t OutputIdx);

/**
 * \returns whether the compilation output data exist in the local CAS given the
 * index via \p OutputIdx.
 */
CINDEX_LINKAGE bool
clang_experimental_cas_CachedCompilation_isOutputMaterialized(
    CXCASCachedCompilation, size_t OutputIdx);

/**
 * If distributed caching is available it uploads the compilation outputs and
 * the association of key <-> outputs to the distributed cache.
 * This allows separating the task of computing the compilation outputs and
 * storing them in the local cache, from the task of "uploading" them.
 *
 * \param Ctx opaque value to pass to the callback.
 * \param Callback receives a \c CXError if an error occurred. The error will be
 * NULL if the call was successful or cancelled. The error should be disposed
 * via \c clang_Error_dispose.
 * \param[out] OutToken if non-null receives a \c CXCASCancellationToken that
 * can be used to cancel the call using
 * \c clang_experimental_cas_CancellationToken_cancel. The object should be
 * disposed using \c clang_experimental_cas_CancellationToken_dispose.
 */
CINDEX_LINKAGE void clang_experimental_cas_CachedCompilation_makeGlobal(
    CXCASCachedCompilation, void *Ctx, void (*Callback)(void *Ctx, CXError),
    CXCASCancellationToken *OutToken);

/**
 * Replays a cached compilation by writing the cached outputs to the filesystem
 * and/or stderr based on the given compilation arguments.
 *
 * \param argc number of compilation arguments.
 * \param argv array of compilation arguments.
 * \param WorkingDirectory working directory to use, can be NULL.
 * \param reserved for future use, caller must pass NULL.
 * \param[out] OutError The error object to pass back to client (if any).
 * If non-null the object must be disposed using \c clang_Error_dispose.
 *
 * \returns a \c CXCASReplayResult object or NULL if an error occurred or a
 * compilation output was not found in the CAS. The object should be disposed
 * via \c clang_experimental_cas_ReplayResult_dispose.
 */
CINDEX_LINKAGE CXCASReplayResult clang_experimental_cas_replayCompilation(
    CXCASCachedCompilation, int argc, const char *const *argv,
    const char *WorkingDirectory, void *reserved, CXError *OutError);

/**
 * Dispose of a \c CXCASReplayResult object.
 */
CINDEX_LINKAGE void
    clang_experimental_cas_ReplayResult_dispose(CXCASReplayResult);

/**
 * Get the diagnostic text of a replayed cached compilation.
 */
CINDEX_LINKAGE
CXString clang_experimental_cas_ReplayResult_getStderr(CXCASReplayResult);

/**
 * Cancel an asynchronous CAS-related action.
 */
CINDEX_LINKAGE void
    clang_experimental_cas_CancellationToken_cancel(CXCASCancellationToken);

/**
 * Dispose of a \c CXCASCancellationToken object.
 */
CINDEX_LINKAGE void
    clang_experimental_cas_CancellationToken_dispose(CXCASCancellationToken);

/**
 * Dispose of a \c CXCASObjectStore object.
 */
CINDEX_LINKAGE void
clang_experimental_cas_ObjectStore_dispose(CXCASObjectStore CAS);

/**
 * Dispose of a \c CXCASActionCache object.
 */
CINDEX_LINKAGE void
clang_experimental_cas_ActionCache_dispose(CXCASActionCache Cache);

/**
 * Gets or creates a persistent on-disk CAS object store at \p Path.
 * Deprecated, use \p clang_experimental_cas_Databases_create() instead.
 *
 * \param Path The path to locate the object store.
 * \param[out] Error The error string to pass back to client (if any).
 *
 * \returns The resulting object store, or null if there was an error.
 */
CINDEX_DEPRECATED CINDEX_LINKAGE CXCASObjectStore
clang_experimental_cas_OnDiskObjectStore_create(const char *Path,
                                                CXString *Error);

/**
 * Gets or creates a persistent on-disk action cache at \p Path.
 * Deprecated, use \p clang_experimental_cas_Databases_create() instead.
 *
 * \param Path The path to locate the object store.
 * \param[out] Error The error string to pass back to client (if any).
 *
 * \returns The resulting object store, or null if there was an error.
 */
CINDEX_DEPRECATED CINDEX_LINKAGE CXCASActionCache
clang_experimental_cas_OnDiskActionCache_create(const char *Path,
                                                CXString *Error);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif // LLVM_CLANG_C_CAS_H
