# Configure the Features.inc file.
if (NOT DEFINED CLANGD_BUILD_XPC)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CLANGD_BUILD_XPC_DEFAULT ON)
  else ()
    set(CLANGD_BUILD_XPC_DEFAULT OFF)
  endif ()

  llvm_canonicalize_cmake_booleans(CLANGD_BUILD_XPC_DEFAULT)

  set(CLANGD_BUILD_XPC ${CLANGD_BUILD_XPC_DEFAULT} CACHE BOOL "Build XPC Support For Clangd." FORCE)
  unset(CLANGD_BUILD_XPC_DEFAULT)
endif ()

llvm_canonicalize_cmake_booleans(CLANGD_BUILD_XPC)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Features.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/Features.inc
)

set(LLVM_LINK_COMPONENTS
  Support
  AllTargetsInfos
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
  CodeComplete.cpp
  CodeCompletionStrings.cpp
  CompileCommands.cpp
  Compiler.cpp
  Context.cpp
  Diagnostics.cpp
  DraftStore.cpp
  ExpectedTypes.cpp
  FindSymbols.cpp
  FindTarget.cpp
  FileDistance.cpp
  Format.cpp
  FS.cpp
  FSProvider.cpp
  FormattedString.cpp
  FuzzyMatch.cpp
  GlobalCompilationDatabase.cpp
  Headers.cpp
  HeaderSourceSwitch.cpp
  Hover.cpp
  IncludeFixer.cpp
  JSONTransport.cpp
  Logger.cpp
  Protocol.cpp
  Quality.cpp
  ParsedAST.cpp
  Preamble.cpp
  RIFF.cpp
  Selection.cpp
  SemanticHighlighting.cpp
  SemanticSelection.cpp
  Shutdown.cpp
  SourceCode.cpp
  QueryDriverDatabase.cpp
  Threading.cpp
  Trace.cpp
  TUScheduler.cpp
  URI.cpp
  XRefs.cpp

  index/Background.cpp
  index/BackgroundIndexLoader.cpp
  index/BackgroundIndexStorage.cpp
  index/BackgroundQueue.cpp
  index/BackgroundRebuild.cpp
  index/CanonicalIncludes.cpp
  index/FileIndex.cpp
  index/Index.cpp
  index/IndexAction.cpp
  index/MemIndex.cpp
  index/Merge.cpp
  index/Ref.cpp
  index/Relation.cpp
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

  refactor/Rename.cpp
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
  clangTooling
  clangToolingCore
  clangToolingInclusions
  clangToolingRefactoring
  clangToolingSyntax
  ${LLVM_PTHREAD_LIB}
  ${CLANGD_ATOMIC_LIB}
  ${ALL_CLANG_TIDY_CHECKS}
  )

add_subdirectory(refactor/tweaks)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
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

if(CLANG_INCLUDE_TESTS)
add_subdirectory(test)
add_subdirectory(unittests)
endif()
