# Configure the Features.inc file.
llvm_canonicalize_cmake_booleans(
  CLANGD_BUILD_XPC)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Features.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/Features.inc
)

set(LLVM_LINK_COMPONENTS
  Support
  )

if(CLANG_BUILT_STANDALONE)
  # needed to get HAVE_CXX_ATOMICS64_WITHOUT_LIB defined
  include(CheckAtomic)
endif()

set(CLANGD_ATOMIC_LIB "")
if(NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB)
  list(APPEND CLANGD_ATOMIC_LIB "atomic")
endif()

add_clang_library(clangDaemon
  AST.cpp
  Cancellation.cpp
  ClangdLSPServer.cpp
  ClangdServer.cpp
  ClangdUnit.cpp
  CodeComplete.cpp
  CodeCompletionStrings.cpp
  Compiler.cpp
  Context.cpp
  Diagnostics.cpp
  DraftStore.cpp
  ExpectedTypes.cpp
  FindSymbols.cpp
  FileDistance.cpp
  FS.cpp
  FSProvider.cpp
  FuzzyMatch.cpp
  GlobalCompilationDatabase.cpp
  Headers.cpp
  IncludeFixer.cpp
  JSONTransport.cpp
  Logger.cpp
  Protocol.cpp
  Quality.cpp
  RIFF.cpp
  Selection.cpp
  SourceCode.cpp
  Threading.cpp
  Trace.cpp
  TUScheduler.cpp
  URI.cpp
  XRefs.cpp

  index/Background.cpp
  index/BackgroundIndexStorage.cpp
  index/CanonicalIncludes.cpp
  index/FileIndex.cpp
  index/Index.cpp
  index/IndexAction.cpp
  index/MemIndex.cpp
  index/Merge.cpp
  index/Ref.cpp
  index/Serialization.cpp
  index/Symbol.cpp
  index/SymbolCollector.cpp
  index/SymbolID.cpp
  index/SymbolLocation.cpp
  index/SymbolOrigin.cpp
  index/YAMLSerialization.cpp

  index/dex/Dex.cpp
  index/dex/Iterator.cpp
  index/dex/PostingList.cpp
  index/dex/Trigram.cpp

  refactor/Tweak.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangDriver
  clangFormat
  clangFrontend
  clangIndex
  clangLex
  clangSema
  clangSerialization
  clangTidy
  clangTidyAndroidModule
  clangTidyAbseilModule
  clangTidyBoostModule
  clangTidyBugproneModule
  clangTidyCERTModule
  clangTidyCppCoreGuidelinesModule
  clangTidyFuchsiaModule
  clangTidyGoogleModule
  clangTidyHICPPModule
  clangTidyLLVMModule
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyObjCModule
  clangTidyPerformanceModule
  clangTidyPortabilityModule
  clangTidyReadabilityModule
  clangTidyZirconModule
  clangTooling
  clangToolingCore
  clangToolingInclusions
  clangToolingRefactor
  ${LLVM_PTHREAD_LIB}
  ${CLANGD_ATOMIC_LIB}
  )

add_subdirectory(refactor/tweaks)
if (LINUX)
  # FIXME: Make fuzzer not use linux-specific APIs, build it everywhere.
  add_subdirectory(fuzzer)
endif()
add_subdirectory(tool)
add_subdirectory(indexer)
add_subdirectory(index/dex/dexp)

if (LLVM_INCLUDE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if ( CLANGD_BUILD_XPC )
  add_subdirectory(xpc)
endif ()
