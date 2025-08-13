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
#include "clang-c/CAS.h"
#include "clang-c/CXDiagnostic.h"
#include "clang-c/CXErrorCode.h"
#include "clang-c/CXString.h"
#include "clang-c/Platform.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup SCAN_DEPS Dependency scanning service.
 * @{
 */

/**
 * An output file kind needed by module dependencies.
 */
typedef enum {
  CXOutputKind_ModuleFile = 0,
  CXOutputKind_Dependencies = 1,
  CXOutputKind_DependenciesTarget = 2,
  CXOutputKind_SerializedDiagnostics = 3,
} CXOutputKind;

/**
 * Object encapsulating instance of a dependency scanner service.
 *
 * The dependency scanner service is a global instance that owns the
 * global cache and other global state that's shared between the dependency
 * scanner workers. The service APIs are thread safe.
 *
 * The service aims to provide a consistent view of file content throughout
 * its lifetime. A client that wants to see changes to file content should
 * create a new service at the time. For example, a build system might use
 * one service for each build.
 *
 * TODO: Consider using DirectoryWatcher to get notified about file changes
 * and adding an API that allows clients to invalidate changed files. This
 * could allow a build system to reuse a single service between builds.
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
 * Options used to construct a \c CXDependencyScannerService.
 */
typedef struct CXOpaqueDependencyScannerServiceOptions
    *CXDependencyScannerServiceOptions;

/**
 * Creates a default set of service options.
 * Must be disposed with \c
 * clang_experimental_DependencyScannerServiceOptions_dispose.
 */
CINDEX_LINKAGE CXDependencyScannerServiceOptions
clang_experimental_DependencyScannerServiceOptions_create(void);

/**
 * Dispose of a \c CXDependencyScannerServiceOptions object.
 */
CINDEX_LINKAGE void clang_experimental_DependencyScannerServiceOptions_dispose(
    CXDependencyScannerServiceOptions);

/**
 * Specify a \c CXDependencyMode in the given options.
 */
CINDEX_LINKAGE void
clang_experimental_DependencyScannerServiceOptions_setDependencyMode(
    CXDependencyScannerServiceOptions Opts, CXDependencyMode Mode);

/**
 * Specify the object store and action cache databases in the given options.
 * With this set, the scanner will produce cached commands.
 */
CINDEX_LINKAGE void
clang_experimental_DependencyScannerServiceOptions_setCASDatabases(
    CXDependencyScannerServiceOptions Opts, CXCASDatabases);

/**
 * Specify the specific CAS options for the scanner to use for the produced
 * compiler arguments.
 */
CINDEX_LINKAGE void
clang_experimental_DependencyScannerServiceOptions_setCASOptions(
    CXDependencyScannerServiceOptions Opts, CXCASOptions);

/**
 * Set the working directory optimization option.
 * The dependency scanner service option Opts will indicate to the scanner that
 * the current working directory can or cannot be ignored when computing the
 * pcms' context hashes. The scanner will then determine if it is safe to
 * optimize each module and act accordingly.
 *
 * \param Value If it is non zero, the option is on. Otherwise the
 * option is off.
 */
CINDEX_LINKAGE void
clang_experimental_DependencyScannerServiceOptions_setCWDOptimization(
    CXDependencyScannerServiceOptions Opts, int Value);

/**
 * Specify a \c CXCASObjectStore in the given options. If an object store and
 * action cache are available, the scanner will produce cached commands.
 * Deprecated, use
 * \p clang_experimental_DependencyScannerServiceOptions_setCASDatabases()
 * instead.
 */
CINDEX_DEPRECATED CINDEX_LINKAGE void
clang_experimental_DependencyScannerServiceOptions_setObjectStore(
    CXDependencyScannerServiceOptions Opts, CXCASObjectStore CAS);

/**
 * Specify a \c CXCASActionCache in the given options. If an object store and
 * action cache are available, the scanner will produce cached commands.
 * Deprecated, use
 * \p clang_experimental_DependencyScannerServiceOptions_setCASDatabases()
 * instead.
 */
CINDEX_DEPRECATED CINDEX_LINKAGE void
clang_experimental_DependencyScannerServiceOptions_setActionCache(
    CXDependencyScannerServiceOptions Opts, CXCASActionCache Cache);

/**
 * Create a \c CXDependencyScannerService object.
 * Must be disposed with \c
 * clang_experimental_DependencyScannerService_dispose_v0().
 */
CINDEX_LINKAGE CXDependencyScannerService
clang_experimental_DependencyScannerService_create_v1(
    CXDependencyScannerServiceOptions Opts);

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
 * A callback that is called to determine the paths of output files for each
 * module dependency. The ModuleFile (pcm) path mapping is mandatory.
 *
 * \param Context the MLOContext that was passed to
 *         \c clang_experimental_DependencyScannerWorkerScanSettings_create().
 * \param ModuleName the name of the dependent module.
 * \param ContextHash the context hash of the dependent module.
 *                    See \c clang_experimental_DepGraphModule_getContextHash().
 * \param OutputKind the kind of module output to lookup.
 * \param[out] Output the output path(s) or name, whose total size must be <=
 *                    \p MaxLen. In the case of multiple outputs of the same
 *                    kind, this can be a null-separated list.
 * \param MaxLen the maximum size of Output.
 *
 * \returns the actual length of Output. If the return value is > \p MaxLen,
 *          the callback will be repeated with a larger buffer.
 */
typedef size_t CXModuleLookupOutputCallback(void *Context,
                                            const char *ModuleName,
                                            const char *ContextHash,
                                            CXOutputKind OutputKind,
                                            char *Output, size_t MaxLen);

/**
 * Output of \c clang_experimental_DependencyScannerWorker_getDepGraph.
 */
typedef struct CXOpaqueDepGraph *CXDepGraph;

/**
 * An individual module dependency that is part of an overall compilation
 * \c CXDepGraph.
 */
typedef struct CXOpaqueDepGraphModule *CXDepGraphModule;

/**
 * An individual command-line invocation that is part of an overall compilation
 * \c CXDepGraph.
 */
typedef struct CXOpaqueDepGraphTUCommand *CXDepGraphTUCommand;

/**
 * Settings to use for the
 * \c clang_experimental_DependencyScannerWorker_getDepGraph action.
 */
typedef struct CXOpaqueDependencyScannerWorkerScanSettings
    *CXDependencyScannerWorkerScanSettings;

/**
 * A callback that is called to determine the paths of output files for each
 * module dependency. The ModuleFile (pcm) path mapping is mandatory.
 *
 * \param Context the MLOContext that was passed to
 *         \c clang_experimental_DependencyScannerWorker_getFileDependencies_vX.
 * \param CXDepMod the ModuleDep that represents the dependent module.
 * \param OutputKind the kind of module output to lookup.
 * \param[out] Output the output path(s) or name, whose total size must be <=
 *                    \p MaxLen. In the case of multiple outputs of the same
 *                    kind, this can be a null-separated list.
 * \param MaxLen the maximum size of Output.
 *
 * \returns the actual length of Output. If the return value is > \p MaxLen,
 *          the callback will be repeated with a larger buffer.
 */
typedef size_t CXModuleLookupOutputCallback_v2(void *Context,
                                               CXDepGraphModule CXDepMod,
                                               CXOutputKind OutputKind,
                                               char *Output, size_t MaxLen);

/**
 * Creates a set of settings for
 * \c clang_experimental_DependencyScannerWorker_getDepGraph action.
 * Must be disposed with
 * \c clang_experimental_DependencyScannerWorkerScanSettings_dispose.
 * Memory for settings is not copied. Any provided pointers must be valid until
 * the call to \c clang_experimental_DependencyScannerWorker_getDepGraph.
 *
 * \param argc the number of compiler invocation arguments (including argv[0]).
 * \param argv the compiler driver invocation arguments (including argv[0]).
 * \param ModuleName If non-null, the dependencies of the named module are
 *                   returned. Otherwise, the dependencies of the whole
 *                   translation unit are returned.
 * \param WorkingDirectory the directory in which the invocation runs.
 * \param MLOContext the context that will be passed to \c MLO each time it is
 *                   called.
 * \param MLO a callback that is called to determine the paths of output files
 *            for each module dependency. This may receive the same module on
 *            different workers. This should be NULL if
 *            \c clang_experimental_DependencyScannerService_create_v1 was
 *            called with \c CXDependencyMode_Flat. This callback will be called
 *            on the same thread that called \c
 *            clang_experimental_DependencyScannerWorker_getDepGraph.
 */
CINDEX_LINKAGE CXDependencyScannerWorkerScanSettings
clang_experimental_DependencyScannerWorkerScanSettings_create(
    int argc, const char *const *argv, const char *ModuleName,
    const char *WorkingDirectory, void *MLOContext,
    CXModuleLookupOutputCallback *MLO);

/**
 * Override the ModuleLookupOutputCallback with \c
 * CXModuleLookupOutputCallback_v2 in the scanner settings for queries of module
 * dependencies. This is required for handling output paths of modules that
 * depend on attributes encoded in
 * \c CXDepGraphModule.
 *
 * \param Settings object created via
 *     \c clang_experimental_DependencyScannerWorkerScanSettings_create.
 * \param MLO a callback that is called to determine the paths of output files
 *            for each module dependency. This may receive the same module on
 *            different workers.
 */
CINDEX_LINKAGE void
clang_experimental_DependencyScannerWorkerScanSettings_setModuleLookupCallback(
    CXDependencyScannerWorkerScanSettings, CXModuleLookupOutputCallback_v2 *);

/**
 * Dispose of a \c CXDependencyScannerWorkerScanSettings object.
 */
CINDEX_LINKAGE void
    clang_experimental_DependencyScannerWorkerScanSettings_dispose(
        CXDependencyScannerWorkerScanSettings);

/**
 * Produces the dependency graph for a particular compiler invocation.
 *
 * \param Settings object created via
 *     \c clang_experimental_DependencyScannerWorkerScanSettings_create.
 * \param [out] Out A non-NULL pointer to store the resulting dependencies. The
 *                  output must be freed by calling
 *                  \c clang_experimental_DepGraph_dispose.
 *
 * \returns \c CXError_Success on success; otherwise a non-zero \c CXErrorCode
 * indicating the kind of error. When returning \c CXError_Failure there will
 * be a \c CXDepGraph object on \p Out that can be used to get diagnostics via
 * \c clang_experimental_DepGraph_getDiagnostics.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_experimental_DependencyScannerWorker_getDepGraph(
    CXDependencyScannerWorker, CXDependencyScannerWorkerScanSettings Settings,
    CXDepGraph *Out);

/**
 * Dispose of a \c CXDepGraph object.
 */
CINDEX_LINKAGE void clang_experimental_DepGraph_dispose(CXDepGraph);

/**
 * \returns the number of \c CXDepGraphModule objects in the graph.
 */
CINDEX_LINKAGE size_t clang_experimental_DepGraph_getNumModules(CXDepGraph);

/**
 * \returns the \c CXDepGraphModule object at the given \p Index.
 *
 * The \c CXDepGraphModule object is only valid to use while \c CXDepGraph is
 * valid. Must be disposed with \c clang_experimental_DepGraphModule_dispose.
 */
CINDEX_LINKAGE CXDepGraphModule
clang_experimental_DepGraph_getModule(CXDepGraph, size_t Index);

/**
 * \returns the \c CXDepGraphModule object at the given \p Index in
 * a topologically sorted list.
 *
 * The \c CXDepGraphModule object is only valid to use while \c CXDepGraph is
 * valid. Must be disposed with \c clang_experimental_DepGraphModule_dispose.
 */
CINDEX_LINKAGE CXDepGraphModule
clang_experimental_DepGraph_getModuleTopological(CXDepGraph, size_t Index);

CINDEX_LINKAGE void clang_experimental_DepGraphModule_dispose(CXDepGraphModule);

/**
 * \returns the name of the module. This may include `:` for C++20 module
 * partitions, or a header-name for C++20 header units.
 *
 * The string is only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE
const char *clang_experimental_DepGraphModule_getName(CXDepGraphModule);

/**
 * \returns the context hash of a module represents the set of compiler options
 * that may make one version of a module incompatible from another. This
 * includes things like language mode, predefined macros, header search paths,
 * etc...
 *
 * Modules with the same name but a different \c ContextHash should be treated
 * as separate modules for the purpose of a build.
 *
 * The string is only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE
const char *clang_experimental_DepGraphModule_getContextHash(CXDepGraphModule);

/**
 * \returns the path to the modulemap file which defines this module. If there's
 * no modulemap (e.g. for a C++ module) returns \c NULL.
 *
 * This can be used to explicitly build this module. This file will
 * additionally appear in \c FileDeps as a dependency.
 *
 * The string is only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE const char *
    clang_experimental_DepGraphModule_getModuleMapPath(CXDepGraphModule);

/**
 * \returns the list of files which this module directly depends on.
 *
 * If any of these change then the module needs to be rebuilt.
 *
 * The strings are only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE CXCStringArray
    clang_experimental_DepGraphModule_getFileDeps(CXDepGraphModule);

/**
 * \returns the list of modules which this module direct depends on.
 *
 * This does include the context hash. The format is
 * `<module-name>:<context-hash>`
 *
 * The strings are only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE CXCStringArray
    clang_experimental_DepGraphModule_getModuleDeps(CXDepGraphModule);

/**
 * \returns whether the provided module is comprised of
 * inputs that resolve into stable directories.
 */
CINDEX_LINKAGE bool
    clang_experimental_DepGraphModule_isInStableDirs(CXDepGraphModule);

/**
 * \returns the canonical command line to build this module.
 *
 * The strings are only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE CXCStringArray
    clang_experimental_DepGraphModule_getBuildArguments(CXDepGraphModule);

/**
 * @returns the CASID of the filesystem root for this module, if any.
 *
 * The string is only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE const char *
    clang_experimental_DepGraphModule_getFileSystemRootID(CXDepGraphModule);

/**
 * @returns the CASID of the include-tree for this module, if any.
 *
 * The string is only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE const char *
    clang_experimental_DepGraphModule_getIncludeTreeID(CXDepGraphModule);

/**
 * \returns the \c ActionCache key for this module, if any.
 *
 * The string is only valid to use while the \c CXDepGraphModule object is
 * valid.
 */
CINDEX_LINKAGE
const char *clang_experimental_DepGraphModule_getCacheKey(CXDepGraphModule);

/**
 * \returns 1 if the scanner ignores the current working directory when
 * computing the module's context hash. Otherwise returns 0.
 */
CINDEX_LINKAGE
int clang_experimental_DepGraphModule_isCWDIgnored(CXDepGraphModule);

/**
 * \returns the number \c CXDepGraphTUCommand objects in the graph.
 */
CINDEX_LINKAGE size_t clang_experimental_DepGraph_getNumTUCommands(CXDepGraph);

/**
 * \returns the \c CXDepGraphTUCommand object at the given \p Index.
 *
 * The \c CXDepGraphTUCommand object is only valid to use while \c CXDepGraph is
 * valid. Must be disposed with \c clang_experimental_DepGraphTUCommand_dispose.
 */
CINDEX_LINKAGE CXDepGraphTUCommand
clang_experimental_DepGraph_getTUCommand(CXDepGraph, size_t Index);

/**
 * Dispose of a \c CXDepGraphTUCommand object.
 */
CINDEX_LINKAGE void
    clang_experimental_DepGraphTUCommand_dispose(CXDepGraphTUCommand);

/**
 * \returns the executable name for the command.
 *
 * The string is only valid to use while the \c CXDepGraphTUCommand object is
 * valid.
 */
CINDEX_LINKAGE const char *
    clang_experimental_DepGraphTUCommand_getExecutable(CXDepGraphTUCommand);

/**
 * \returns the canonical command line to build this translation unit.
 *
 * The strings are only valid to use while the \c CXDepGraphTUCommand object is
 * valid.
 */
CINDEX_LINKAGE CXCStringArray
    clang_experimental_DepGraphTUCommand_getBuildArguments(CXDepGraphTUCommand);

/**
 * \returns the \c ActionCache key for this translation unit, if any.
 *
 * The string is only valid to use while the \c CXDepGraphTUCommand object is
 * valid.
 */
CINDEX_LINKAGE const char *
    clang_experimental_DepGraphTUCommand_getCacheKey(CXDepGraphTUCommand);

/**
 * \returns the list of files which this translation unit directly depends on.
 *
 * The strings are only valid to use while the \c CXDepGraph object is valid.
 */
CINDEX_LINKAGE
CXCStringArray clang_experimental_DepGraph_getTUFileDeps(CXDepGraph);

/**
 * \returns the list of modules which this translation unit direct depends on.
 *
 * This does include the context hash. The format is
 * `<module-name>:<context-hash>`
 *
 * The strings are only valid to use while the \c CXDepGraph object is valid.
 */
CINDEX_LINKAGE
CXCStringArray clang_experimental_DepGraph_getTUModuleDeps(CXDepGraph);

/**
 * @returns the CASID of the filesystem root for this TU, if any.
 *
 * This string is only valid to use while the \c CXDepGraph object is valid.
 */
CINDEX_LINKAGE
const char *clang_experimental_DepGraph_getTUFileSystemRootID(CXDepGraph);

/**
 * @returns the CASID of the include-tree for this TU, if any.
 *
 * The string is only valid to use while the \c CXDepGraph object is valid.
 */
CINDEX_LINKAGE
const char *clang_experimental_DepGraph_getTUIncludeTreeID(CXDepGraph);

/**
 * \returns the context hash of the C++20 module this translation unit exports.
 *
 * If the translation unit is not a module then this is empty.
 *
 * The string is only valid to use while the \c CXDepGraph object is valid.
 */
CINDEX_LINKAGE
const char *clang_experimental_DepGraph_getTUContextHash(CXDepGraph);

/**
 * \returns The diagnostics emitted during scanning. These must be always freed
 * by calling \c clang_disposeDiagnosticSet.
 */
CINDEX_LINKAGE
CXDiagnosticSet clang_experimental_DepGraph_getDiagnostics(CXDepGraph);

/**
 * The kind of scanning file system cache out-of-date entries.
 */
typedef enum {
  /**
   * The entry is negatively stat cached (which indicates the file did not exist
   * the first time it was looked up during scanning), but the cached file
   * exists on the underlying file system.
   */
  NegativelyCached,

  /**
   * The entry indicates that for the cached file, its cached size
   * is different from its size reported by the underlying file system.
   */
  SizeChanged
} CXDepScanFSCacheOutOfDateKind;

/**
 * The opaque object that contains the scanning file system cache's out-of-date
 * entires.
 */
typedef struct CXOpaqueDepScanFSOutOfDateEntrySet *CXDepScanFSOutOfDateEntrySet;

/**
 * The opaque object that represents a single scanning file system cache's out-
 * of-date entry.
 */
typedef struct CXOpaqueDepScanFSOutOfDateEntry *CXDepScanFSOutOfDateEntry;

/**
 * Returns all the file system cache out-of-date entries given a
 * \c CXDependencyScannerService .
 *
 * This function is intended to be called when the build has finished,
 * and the \c CXDependencyScannerService instance is about to be disposed.
 *
 * The \c CXDependencyScannerService instance owns the strings used
 * by the out-of-date entries and should be disposed after the
 * out-of-date entries are used and disposed.
 */
CINDEX_LINKAGE CXDepScanFSOutOfDateEntrySet
clang_experimental_DependencyScannerService_getFSCacheOutOfDateEntrySet(
    CXDependencyScannerService S);

/**
 * Returns the number of out-of-date entries contained in a
 * \c CXDepScanFSOutOfDateEntrySet .
 */
CINDEX_LINKAGE size_t
clang_experimental_DepScanFSCacheOutOfDateEntrySet_getNumOfEntries(
    CXDepScanFSOutOfDateEntrySet Entries);

/**
 * Returns the out-of-date entry at offset \p Idx of the \c
 * CXDepScanFSOutOfDateEntrySet instance.
 */
CINDEX_LINKAGE CXDepScanFSOutOfDateEntry
clang_experimental_DepScanFSCacheOutOfDateEntrySet_getEntry(
    CXDepScanFSOutOfDateEntrySet Entries, size_t Idx);

/**
 * Given an instance of \c CXDepScanFSOutOfDateEntry, returns its Kind.
 */
CINDEX_LINKAGE CXDepScanFSCacheOutOfDateKind
clang_experimental_DepScanFSCacheOutOfDateEntry_getKind(
    CXDepScanFSOutOfDateEntry Entry);

/**
 * Given an instance of \c CXDepScanFSOutOfDateEntry, returns the path.
 */
CINDEX_LINKAGE CXString clang_experimental_DepScanFSCacheOutOfDateEntry_getPath(
    CXDepScanFSOutOfDateEntry Entry);

/**
 * Given an instance of \c CXDepScanFSOutOfDateEntry of kind SizeChanged,
 * returns the cached size.
 */
CINDEX_LINKAGE uint64_t
clang_experimental_DepScanFSCacheOutOfDateEntry_getCachedSize(
    CXDepScanFSOutOfDateEntry Entry);

/**
 * Given an instance of \c CXDepScanFSOutOfDateEntry of kind SizeChanged,
 * returns the actual size on the underlying file system.
 */
CINDEX_LINKAGE uint64_t
clang_experimental_DepScanFSCacheOutOfDateEntry_getActualSize(
    CXDepScanFSOutOfDateEntry Entry);

/**
 * Dispose the \c CXDepScanFSOutOfDateEntrySet instance.
 */
CINDEX_LINKAGE void
clang_experimental_DepScanFSCacheOutOfDateEntrySet_disposeSet(
    CXDepScanFSOutOfDateEntrySet Entries);

/**
 * Options used to generate a reproducer.
 */
typedef struct CXOpaqueDependencyScannerReproducerOptions
    *CXDependencyScannerReproducerOptions;

/**
 * Creates a set of settings for
 * \c clang_experimental_DependencyScanner_generateReproducer action.
 * Must be disposed with
 * \c clang_experimental_DependencyScannerReproducerOptions_dispose.
 *
 * \param argc the number of compiler invocation arguments (including argv[0]).
 * \param argv the compiler driver invocation arguments (including argv[0]).
 * \param ModuleName If non-NULL, reproduce building the named module and all
 *                   the intermediate modules. Otherwise, reproduce building
 *                   the whole translation unit.
 * \param WorkingDirectory the directory in which the invocation runs.
 * \param ReproducerLocation the directory where to store the reproducer files.
 *                           If NULL, use a temporary location.
 * \param UseUniqueReproducerName if reproducer files should have unique names
 *                                to avoid collisions with existing files.
 */
CINDEX_LINKAGE CXDependencyScannerReproducerOptions
clang_experimental_DependencyScannerReproducerOptions_create(
    int argc, const char *const *argv, const char *ModuleName,
    const char *WorkingDirectory, const char *ReproducerLocation,
    bool UseUniqueReproducerName);

CINDEX_LINKAGE void
    clang_experimental_DependencyScannerReproducerOptions_dispose(
        CXDependencyScannerReproducerOptions);

/**
 * Generates a reproducer to compile a requested file with required modules.
 *
 * Here the reproducer means the required input data with the commands to
 * compile intermediate modules and a requested file. Required intermediate
 * modules and the order of their compilation are determined by the function
 * and don't need to be provided.
 *
 * \param CXOptions object created via
 *     \c clang_experimental_DependencyScannerReproducerOptions_create.
 * \param [out] MessageOut A pointer to store the human-readable message
 *                         describing the result of the operation. If non-NULL,
 *                         owned and should be disposed by the caller.
 *
 * \returns \c CXError_Success on success; otherwise a non-zero \c CXErrorCode
 * indicating the kind of error. \p MessageOut is guaranteed to be populated
 * for a success case but is allowed to be empty when encountered an error.
 */
CINDEX_LINKAGE enum CXErrorCode
clang_experimental_DependencyScanner_generateReproducer(
    CXDependencyScannerReproducerOptions CXOptions, CXString *MessageOut);

/**
 * An opaque object that contains a \c CXDepGraphModule 's list of link
 * libraries.
 */
typedef struct CXOpaqueDepGraphModuleLinkLibrarySet
    *CXDepGraphModuleLinkLibrarySet;

/**
 * An opaque object that contains information about a link library.
 */
typedef struct CXOpaqueDepGraphModuleLinkLibrary *CXDepGraphModuleLinkLibrary;

/**
 * Get the set of link libraries given a \c CXDepGraphModule instance.
 * The returned set is a pointer into memory that the \c CXDepGraphModule
 * instance owns. Therefore the set does not need to be disposed, and it has
 * the same lifetime as the \c CXDepGraphModule instance.
 */
CINDEX_LINKAGE CXDepGraphModuleLinkLibrarySet
    clang_experimental_DepGraphModule_getLinkLibrarySet(CXDepGraphModule);

/**
 * Get the size of \c CXDepGraphModuleLinkLibrarySet .
 */
CINDEX_LINKAGE size_t clang_experimental_DepGraphModuleLinkLibrarySet_getSize(
    CXDepGraphModuleLinkLibrarySet);

/**
 * Retrieve the \c CXDepGraphModuleLinkLibrary instance at index \p Idx from the
 * \c CXDepGraphModuleLinkLibrarySet instance. The returned object is owned
 * by the \c CXDepGraphModule instance where the
 * \c CXDepGraphModuleLinkLibrarySet was retrieved from, and has the same
 * lifetime as the \c CXDepGraphModule instance.
 */
CINDEX_LINKAGE CXDepGraphModuleLinkLibrary
clang_experimental_DepGraphModuleLinkLibrarySet_getLinkLibrary(
    CXDepGraphModuleLinkLibrarySet, size_t Idx);

/**
 * Get the `Library` string from the \c CXDepGraphModuleLinkLibrary instance.
 * `Library` could be a library name, or an absolute path to a library or a
 * framework, as specified in the corresponding \c CXDepGraphModule instance's
 * modulemap. The returned \c CXString is a reference to a string owned
 * by the \c CXDepGraphModule instance, and has the same lifetime as the
 * the \c CXDepGraphModule instance.
 */
CINDEX_LINKAGE CXString clang_experimental_DepGraphModuleLinkLibrary_getLibrary(
    CXDepGraphModuleLinkLibrary);

/**
 * Returns true if the \c CXDepGraphModuleLinkLibrary is a framework, false
 * otherwise.
 */
CINDEX_LINKAGE bool clang_experimental_DepGraphModuleLinkLibrary_isFramework(
    CXDepGraphModuleLinkLibrary);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif // LLVM_CLANG_C_DEPENDENCIES_H
