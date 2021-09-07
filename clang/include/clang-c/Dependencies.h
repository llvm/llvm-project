/*==-- clang-c/Dependencies.h - Dependency Discovery C Interface --*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides a dependency discovery interface similar to           *|
|* clang-scan-deps.                                                           *|
|*                                                                            *|
|* An example of its usage is available in c-index-test/core_main.cpp.        *|
|*                                                                            *|
|* EXPERIMENTAL: These interfaces are experimental and will change. If you    *|
|* use these be prepared for them to change without notice on any commit.     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_CLANG_C_DEPENDENCIES_H
#define LLVM_CLANG_C_DEPENDENCIES_H

#include "clang-c/BuildSystem.h"
#include "clang-c/CXErrorCode.h"
#include "clang-c/CXString.h"
#include "clang-c/Platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup SCAN_DEPS Dependency scanning service.
 * @{
 */

typedef struct {
  CXString Name;
  /**
   * The context hash of a module represents the set of compiler options that
   * may make one version of a module incompatible from another. This includes
   * things like language mode, predefined macros, header search paths, etc...
   *
   * Modules with the same name but a different \c ContextHash should be treated
   * as separate modules for the purpose of a build.
   */
  CXString ContextHash;

  /**
   * The path to the modulemap file which defines this module.
   *
   * This can be used to explicitly build this module. This file will
   * additionally appear in \c FileDeps as a dependency.
   */
  CXString ModuleMapPath;

  /**
   * The list of files which this module directly depends on.
   *
   * If any of these change then the module needs to be rebuilt.
   */
  CXStringSet *FileDeps;

  /**
   * The list of modules which this module direct depends on.
   *
   * This does include the context hash. The format is
   * `<module-name>:<context-hash>`
   */
  CXStringSet *ModuleDeps;

  /**
   * The canonical command-line or additional arguments needed to build this
   * module, excluding arguments containing modules-related paths:
   * "-fmodule-file=", "-o", "-fmodule-map-file=".
   */
  CXStringSet *BuildArguments;
} CXModuleDependency;

typedef struct {
  int Count;
  CXModuleDependency *Modules;
} CXModuleDependencySet;

/**
 * See \c CXModuleDependency for the meaning of these fields, with the addition
 * that they represent only the direct dependencies for \c CXDependencyMode_Full
 * mode.
 */
typedef struct {
  CXString ContextHash;
  CXStringSet *FileDeps;
  CXStringSet *ModuleDeps;
  
  /**
   * Additional arguments to append to the build of this file.
   *
   * This contains things like disabling implicit modules. This does not include
   * the `-fmodule-file=` arguments that are needed.
   */
  CXStringSet *AdditionalArguments;
} CXFileDependencies;

CINDEX_LINKAGE void
clang_experimental_ModuleDependencySet_dispose(CXModuleDependencySet *MD);

CINDEX_LINKAGE void
clang_experimental_FileDependencies_dispose(CXFileDependencies *ID);

/**
 * Object encapsulating instance of a dependency scanner service.
 *
 * The dependency scanner service is a global instance that owns the
 * global cache and other global state that's shared between the dependency
 * scanner workers. The service APIs are thread safe.
 */
typedef struct CXOpaqueDependencyScannerService *CXDependencyScannerService;

/**
 * The mode to report module dependencies in.
 */
typedef enum {
  /**
   * Flatten all module dependencies. This reports the full transitive set of
   * header and module map dependencies needed to do an implicit module build.
   */
  CXDependencyMode_Flat,

  /**
   * Report the full module graph. This reports only the direct dependencies of
   * each file, and calls a callback for each module that is discovered.
   */
  CXDependencyMode_Full,
} CXDependencyMode;

/**
 * Create a \c CXDependencyScannerService object.
 * Must be disposed with \c clang_DependencyScannerService_dispose().
 */
CINDEX_LINKAGE CXDependencyScannerService
clang_experimental_DependencyScannerService_create_v0(CXDependencyMode Format);

/**
 * Dispose of a \c CXDependencyScannerService object.
 *
 * The service object must be disposed of after the workers are disposed of.
 */
CINDEX_LINKAGE void clang_experimental_DependencyScannerService_dispose_v0(
    CXDependencyScannerService);

/**
 * Object encapsulating instance of a dependency scanner worker.
 *
 * The dependency scanner workers are expected to be used in separate worker
 * threads. An individual worker is not thread safe.
 *
 * Operations on a worker are not thread-safe and should only be used from a
 * single thread at a time. They are intended to be used by a single dedicated
 * thread in a thread pool, but they are not inherently pinned to a thread.
 */
typedef struct CXOpaqueDependencyScannerWorker *CXDependencyScannerWorker;

/**
 * Create a \c CXDependencyScannerWorker object.
 * Must be disposed with
 * \c clang_experimental_DependencyScannerWorker_dispose_v0().
 */
CINDEX_LINKAGE CXDependencyScannerWorker
    clang_experimental_DependencyScannerWorker_create_v0(
        CXDependencyScannerService);

CINDEX_LINKAGE void clang_experimental_DependencyScannerWorker_dispose_v0(
    CXDependencyScannerWorker);

/**
 * A callback that is called whenever a module is discovered when in
 * \c CXDependencyMode_Full mode.
 *
 * \param Context the context that was passed to
 *         \c clang_experimental_DependencyScannerWorker_getFileDependencies_v0.
 * \param MDS the list of discovered modules. Must be freed by calling
 *            \c clang_experimental_ModuleDependencySet_dispose.
 */
typedef void CXModuleDiscoveredCallback(void *Context,
                                        CXModuleDependencySet *MDS);

/**
 * Returns the list of file dependencies for a particular compiler invocation.
 *
 * \param argc the number of compiler invocation arguments (including argv[0]).
 * \param argv the compiler invocation arguments (including argv[0]).
 *             the invocation may be a -cc1 clang invocation or a driver
 *             invocation.
 * \param WorkingDirectory the directory in which the invocation runs.
 * \param MDC a callback that is called whenever a new module is discovered.
 *            This may receive the same module on different workers. This should
 *            be NULL if
 *            \c clang_experimental_DependencyScannerService_create_v0 was
 *            called with \c CXDependencyMode_Flat. This callback will be called
 *            on the same thread that called this function.
 * \param Context the context that will be passed to \c MDC each time it is
 *                called.
 * \param [out] error the error string to pass back to client (if any).
 *
 * \returns A pointer to a CXFileDependencies on success, NULL otherwise. The
 *          CXFileDependencies must be freed by calling
 *          \c clang_experimental_FileDependencies_dispose.
 */
CINDEX_LINKAGE CXFileDependencies *
clang_experimental_DependencyScannerWorker_getFileDependencies_v0(
    CXDependencyScannerWorker Worker, int argc, const char *const *argv,
    const char *WorkingDirectory, CXModuleDiscoveredCallback *MDC,
    void *Context, CXString *error);

/**
 * Same as \c clang_experimental_DependencyScannerWorker_getFileDependencies_v0,
 * but \c BuildArguments of each \c CXModuleDependency passed to \c MDC contains
 * the canonical Clang command line, not just additional arguments.
 */
CINDEX_LINKAGE CXFileDependencies *
clang_experimental_DependencyScannerWorker_getFileDependencies_v1(
    CXDependencyScannerWorker Worker, int argc, const char *const *argv,
    const char *WorkingDirectory, CXModuleDiscoveredCallback *MDC,
    void *Context, CXString *error);

/**
 * Same as \c clang_experimental_DependencyScannerWorker_getFileDependencies_v1,
 * but get the dependencies by module name alone.
 */
CINDEX_LINKAGE CXFileDependencies *
clang_experimental_DependencyScannerWorker_getDependenciesByModuleName_v0(
    CXDependencyScannerWorker Worker, int argc, const char *const *argv,
    const char *ModuleName, const char *WorkingDirectory,
    CXModuleDiscoveredCallback *MDC, void *Context, CXString *error);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif // LLVM_CLANG_C_DEPENDENCIES_H
