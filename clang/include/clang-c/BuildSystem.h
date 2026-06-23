/*==-- clang-c/BuildSystem.h - Utilities for use by build systems -*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides various utilities for use by build systems.           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_CLANG_C_BUILDSYSTEM_H
#define LLVM_CLANG_C_BUILDSYSTEM_H

#include "clang-c/CXErrorCode.h"
#include "clang-c/CXString.h"
#include "clang-c/ExternC.h"
#include "clang-c/Platform.h"
#include <time.h>

LLVM_CLANG_C_EXTERN_C_BEGIN

/**
 * \defgroup BUILD_SYSTEM Build system utilities
 * @{
 */

/**
 * Return the timestamp for use with Clang's
 * \c -fbuild-session-timestamp= option.
 */
CINDEX_LINKAGE unsigned long long clang_getBuildSessionTimestamp(void);

/**
 * Object encapsulating information about overlaying virtual
 * file/directories over the real file system.
 */
typedef struct CXVirtualFileOverlayImpl *CXVirtualFileOverlay;

/**
 * Create a \c CXVirtualFileOverlay object.
 * Must be disposed with \c clang_VirtualFileOverlay_dispose().
 *
 * \param options is reserved, always pass 0.
 */
CINDEX_LINKAGE CXVirtualFileOverlay
clang_VirtualFileOverlay_create(unsigned options);

/**
 * Map an absolute virtual file path to an absolute real one.
 * The virtual path must be canonicalized (not contain "."/"..").
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_addFileMapping(CXVirtualFileOverlay,
                                        const char *virtualPath,
                                        const char *realPath);

/**
 * Set the case sensitivity for the \c CXVirtualFileOverlay object.
 * The \c CXVirtualFileOverlay object is case-sensitive by default, this
 * option can be used to override the default.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_setCaseSensitivity(CXVirtualFileOverlay,
                                            int caseSensitive);

/**
 * Write out the \c CXVirtualFileOverlay object to a char buffer.
 *
 * \param options is reserved, always pass 0.
 * \param out_buffer_ptr pointer to receive the buffer pointer, which should be
 * disposed using \c clang_free().
 * \param out_buffer_size pointer to receive the buffer size.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_VirtualFileOverlay_writeToBuffer(CXVirtualFileOverlay, unsigned options,
                                       char **out_buffer_ptr,
                                       unsigned *out_buffer_size);

/**
 * free memory allocated by libclang, such as the buffer returned by
 * \c CXVirtualFileOverlay() or \c clang_ModuleMapDescriptor_writeToBuffer().
 *
 * \param buffer memory pointer to free.
 */
CINDEX_LINKAGE void clang_free(void *buffer);

/**
 * Dispose a \c CXVirtualFileOverlay object.
 */
CINDEX_LINKAGE void clang_VirtualFileOverlay_dispose(CXVirtualFileOverlay);

/**
 * Object encapsulating information about a module.modulemap file.
 */
typedef struct CXModuleMapDescriptorImpl *CXModuleMapDescriptor;

/**
 * Create a \c CXModuleMapDescriptor object.
 * Must be disposed with \c clang_ModuleMapDescriptor_dispose().
 *
 * \param options is reserved, always pass 0.
 */
CINDEX_LINKAGE CXModuleMapDescriptor
clang_ModuleMapDescriptor_create(unsigned options);

/**
 * Sets the framework module name that the module.modulemap describes.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_ModuleMapDescriptor_setFrameworkModuleName(CXModuleMapDescriptor,
                                                 const char *name);

/**
 * Sets the umbrella header name that the module.modulemap describes.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_ModuleMapDescriptor_setUmbrellaHeader(CXModuleMapDescriptor,
                                            const char *name);

/**
 * Write out the \c CXModuleMapDescriptor object to a char buffer.
 *
 * \param options is reserved, always pass 0.
 * \param out_buffer_ptr pointer to receive the buffer pointer, which should be
 * disposed using \c clang_free().
 * \param out_buffer_size pointer to receive the buffer size.
 * \returns 0 for success, non-zero to indicate an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_ModuleMapDescriptor_writeToBuffer(CXModuleMapDescriptor, unsigned options,
                                       char **out_buffer_ptr,
                                       unsigned *out_buffer_size);

/**
 * Dispose a \c CXModuleMapDescriptor object.
 */
CINDEX_LINKAGE void clang_ModuleMapDescriptor_dispose(CXModuleMapDescriptor);

/**
 * Prune module files in the module cache directory that haven't been accessed
 * in a long time.
 *
 * \param Path the path to the module cache directory.
 *
 * \param PruneInterval the minimum time in seconds between two prune
 * operations. If the timestamp file is newer than this, pruning is skipped.
 *
 * \param PruneAfter the time in seconds after which unused module files are
 * removed.
 *
 */
CINDEX_LINKAGE void clang_ModuleCache_prune(const char *Path,
                                            time_t PruneInterval,
                                            time_t PruneAfter);

/**
 * Callback invoked by \c clang_ModuleCache_pruneWithCallback() once for each
 * file or directory removed from the module cache.
 *
 * \param Path the absolute path of the file or directory that was pruned.
 * The pointer is only valid for the duration of the callback.
 *
 * \param UserData the opaque pointer passed to
 * \c clang_ModuleCache_pruneWithCallback().
 */
typedef void (*CXModuleCachePruneCallback)(const char *Path, void *UserData);

/**
 * Variant of \c clang_ModuleCache_prune() that invokes \p Callback once for
 * each absolute path removed from the cache. This includes pruned \c .pcm
 * files, their companion \c .timestamp files, and any cache subdirectory that
 * becomes empty as a result of pruning.
 *
 * \param Path the path to the module cache directory.
 *
 * \param PruneInterval the minimum time in seconds between two prune
 * operations. If the timestamp file is newer than this, pruning is skipped.
 *
 * \param PruneAfter the time in seconds after which unused module files are
 * removed.
 *
 * \param Callback invoked for each pruned absolute path. May be NULL, in
 * which case this function behaves like \c clang_ModuleCache_prune().
 *
 * \param UserData opaque pointer passed through to \p Callback.
 */
CINDEX_LINKAGE void clang_ModuleCache_pruneWithCallback(
    const char *Path, time_t PruneInterval, time_t PruneAfter,
    CXModuleCachePruneCallback Callback, void *UserData);

/**
 * @}
 */

LLVM_CLANG_C_EXTERN_C_END

#endif /* CLANG_C_BUILD_SYSTEM_H */

