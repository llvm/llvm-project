add_subdirectory(support)

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
  FrontendOpenMP
  )

add_clang_library(clangDaemon
  AST.cpp
  ClangdLSPServer.cpp
  ClangdServer.cpp
  CodeComplete.cpp
  CodeCompletionStrings.cpp
  CollectMacros.cpp
  CompileCommands.cpp
  Compiler.cpp
  ConfigYAML.cpp
  Diagnostics.cpp
  DraftStore.cpp
  ExpectedTypes.cpp
  FindSymbols.cpp
  FindTarget.cpp
  FileDistance.cpp
  Format.cpp
  FS.cpp
  FuzzyMatch.cpp
  GlobalCompilationDatabase.cpp
  Headers.cpp
  HeaderSourceSwitch.cpp
  Hover.cpp
  IncludeFixer.cpp
  JSONTransport.cpp
  PathMapping.cpp
  Protocol.cpp
  Quality.cpp
  ParsedAST.cpp
  Preamble.cpp
  RIFF.cpp
  Selection.cpp
  SemanticHighlighting.cpp
  SemanticSelection.cpp
  SourceCode.cpp
  QueryDriverDatabase.cpp
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
  clangdSupport
  clangTidy
  ${LLVM_PTHREAD_LIB}
  ${ALL_CLANG_TIDY_CHECKS}

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(clangDaemon
  PRIVATE
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
  clangTooling
  clangToolingCore
  clangToolingInclusions
  clangToolingRefactoring
  clangToolingSyntax
  )

add_subdirectory(refactor/tweaks)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  # FIXME: Make fuzzer not use linux-specific APIs, build it everywhere.
  add_subdirectory(fuzzer)
endif()
add_subdirectory(tool)
add_subdirectory(indexer)

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

# FIXME(kirillbobyrev): Document this in the LLVM docs once remote index is stable.
option(CLANGD_ENABLE_REMOTE "Use gRPC library to enable remote index support for Clangd" OFF)
set(GRPC_INSTALL_PATH "" CACHE PATH "Path to gRPC library manual installation.")

if (CLANGD_ENABLE_REMOTE)
  include(FindGRPC)
endif()
add_subdirectory(index/remote)
add_subdirectory(index/dex/dexp)
