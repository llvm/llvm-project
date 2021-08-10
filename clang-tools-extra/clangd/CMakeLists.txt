# This is a no-op for building files in this dir, but is inherited by subdirs.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

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

option(CLANGD_MALLOC_TRIM "Call malloc_trim(3) periodically in Clangd. (only takes effect when using glibc)" ON)
# -DCLANG_TIDY_CHECKS=Off avoids a dependency on clang-tidy, reducing rebuilds.
option(CLANGD_TIDY_CHECKS "Link all clang-tidy checks into clangd" ON)

llvm_canonicalize_cmake_booleans(
  CLANGD_BUILD_XPC
  CLANGD_ENABLE_REMOTE
  ENABLE_GRPC_REFLECTION
  CLANGD_MALLOC_TRIM
  CLANGD_TIDY_CHECKS
  LLVM_ENABLE_ZLIB
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Features.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/Features.inc
)

set(LLVM_LINK_COMPONENTS
  Support
  AllTargetsInfos
  FrontendOpenMP
  Option
  )

include(${CMAKE_CURRENT_SOURCE_DIR}/quality/CompletionModel.cmake)
gen_decision_forest(${CMAKE_CURRENT_SOURCE_DIR}/quality/model CompletionModel clang::clangd::Example)

if(MSVC AND NOT CLANG_CL)
 set_source_files_properties(CompileCommands.cpp PROPERTIES COMPILE_FLAGS -wd4130) # disables C4130: logical operation on address of string constant
endif()

include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/../clang-tidy")

add_clang_library(clangDaemon
  AST.cpp
  ASTSignals.cpp
  ClangdLSPServer.cpp
  ClangdServer.cpp
  CodeComplete.cpp
  CodeCompletionStrings.cpp
  CollectMacros.cpp
  CompileCommands.cpp
  Compiler.cpp
  Config.cpp
  ConfigCompile.cpp
  ConfigProvider.cpp
  ConfigYAML.cpp
  Diagnostics.cpp
  DraftStore.cpp
  DumpAST.cpp
  ExpectedTypes.cpp
  FeatureModule.cpp
  Feature.cpp
  FindSymbols.cpp
  FindTarget.cpp
  FileDistance.cpp
  Format.cpp
  FS.cpp
  FuzzyMatch.cpp
  GlobalCompilationDatabase.cpp
  Headers.cpp
  HeaderSourceSwitch.cpp
  HeuristicResolver.cpp
  Hover.cpp
  IncludeFixer.cpp
  InlayHints.cpp
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
  TidyProvider.cpp
  TUScheduler.cpp
  URI.cpp
  XRefs.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/CompletionModel.cpp

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
  index/ProjectAware.cpp
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

  DEPENDS
  omp_gen
  )

# Include generated CompletionModel headers.
target_include_directories(clangDaemon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
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
  clangToolingSyntax
  )

target_link_libraries(clangDaemon
  PRIVATE
  ${LLVM_PTHREAD_LIB}

  clangTidy

  clangdSupport
  )
if(CLANGD_TIDY_CHECKS)
  target_link_libraries(clangDaemon PRIVATE ${ALL_CLANG_TIDY_CHECKS})
endif()

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

if (CLANGD_ENABLE_REMOTE)
  include(FindGRPC)
endif()

if(CLANG_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()

# FIXME(kirillbobyrev): Document this in the LLVM docs once remote index is stable.
option(CLANGD_ENABLE_REMOTE "Use gRPC library to enable remote index support for Clangd" OFF)
set(GRPC_INSTALL_PATH "" CACHE PATH "Path to gRPC library manual installation.")

add_subdirectory(index/remote)
add_subdirectory(index/dex/dexp)
