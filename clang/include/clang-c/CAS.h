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

#include "clang-c/CXString.h"
#include "clang-c/Platform.h"

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
