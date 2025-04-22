# This file allows users to call find_package(Clang) and pick up our targets.

# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(CLANG_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" REALPATH)
get_filename_component(CLANG_INSTALL_PREFIX "${CLANG_INSTALL_PREFIX}" PATH)
get_filename_component(CLANG_INSTALL_PREFIX "${CLANG_INSTALL_PREFIX}" PATH)
get_filename_component(CLANG_INSTALL_PREFIX "${CLANG_INSTALL_PREFIX}" PATH)
get_filename_component(CLANG_INSTALL_PREFIX "${CLANG_INSTALL_PREFIX}" PATH)

set(LLVM_VERSION 21.0.0)
find_package(LLVM ${LLVM_VERSION} EXACT REQUIRED CONFIG
             HINTS "${CLANG_INSTALL_PREFIX}/lib/cmake/llvm")

set(CLANG_EXPORTED_TARGETS "clang-tblgen;clang-resource-headers;clangBasic;clangAPINotes;clangLex;clangParse;clangAST;clangDynamicASTMatchers;clangASTMatchers;clangCrossTU;clangSema;clangCodeGen;clangAnalysis;clangAnalysisFlowSensitive;clangAnalysisFlowSensitiveModels;clangEdit;clangExtractAPI;clangRewrite;clangDriver;clangSerialization;clangRewriteFrontend;clangFrontend;clangFrontendTool;clangToolingCore;clangToolingInclusions;clangToolingInclusionsStdlib;clangToolingRefactoring;clangToolingASTDiff;clangToolingSyntax;clangDependencyScanning;clangTransformer;clangTooling;clangDirectoryWatcher;clangIndex;clangIndexSerialization;clangInstallAPI;clangStaticAnalyzerCore;clangStaticAnalyzerCheckers;clangStaticAnalyzerFrontend;clangFormat;clangInterpreter;clangSupport;diagtool;clang;clang-format;clangHandleCXX;clangHandleLLVM;clang-linker-wrapper;clang-nvlink-wrapper;clang-offload-packager;clang-offload-bundler;clang-scan-deps;clang-sycl-linker;clang-installapi;clang-repl;clang-refactor;clang-cpp;clang-check;clang-extdef-mapping;clangApplyReplacements;clang-apply-replacements;clangReorderFields;clang-reorder-fields;modularize;clangTidy;clangTidyAndroidModule;clangTidyAbseilModule;clangTidyAlteraModule;clangTidyBoostModule;clangTidyBugproneModule;clangTidyCERTModule;clangTidyConcurrencyModule;clangTidyCppCoreGuidelinesModule;clangTidyDarwinModule;clangTidyFuchsiaModule;clangTidyGoogleModule;clangTidyHICPPModule;clangTidyLinuxKernelModule;clangTidyLLVMModule;clangTidyLLVMLibcModule;clangTidyMiscModule;clangTidyModernizeModule;clangTidyMPIModule;clangTidyObjCModule;clangTidyOpenMPModule;clangTidyPerformanceModule;clangTidyPortabilityModule;clangTidyReadabilityModule;clangTidyZirconModule;clangTidyPlugin;clangTidyMain;clang-tidy;clangTidyUtils;clangChangeNamespace;clang-change-namespace;clangDocSupport;clangDoc;clang-doc;clangIncludeFixer;clangIncludeFixerPlugin;clang-include-fixer;findAllSymbols;find-all-symbols;clangMove;clang-move;clangQuery;clang-query;clangIncludeCleaner;clang-include-cleaner;pp-trace;clangdSupport;clangDaemon;clangDaemonTweaks;clangdMain;clangd;clangdRemoteIndex;libclang;offload-arch")
set(CLANG_CMAKE_DIR "${CLANG_INSTALL_PREFIX}/lib/cmake/clang")
set(CLANG_INCLUDE_DIRS "${CLANG_INSTALL_PREFIX}/include")
set(CLANG_LINK_CLANG_DYLIB "OFF")

# Provide all our library targets to users.
include("${CLANG_CMAKE_DIR}/ClangTargets.cmake")

# By creating clang-tablegen-targets here, subprojects that depend on Clang's
# tablegen-generated headers can always depend on this target whether building
# in-tree with Clang or not.
if(NOT TARGET clang-tablegen-targets)
  add_custom_target(clang-tablegen-targets)
endif()
