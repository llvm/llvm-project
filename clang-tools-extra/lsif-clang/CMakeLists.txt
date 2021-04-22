cmake_policy(SET CMP0048 NEW)

project(ClangLSIF VERSION 0.1)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -g")
set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra")

if (PATH_TO_LLVM)
  set (search_paths
    ${PATH_TO_LLVM}
    ${PATH_TO_LLVM}/cmake
    ${PATH_TO_LLVM}/lib/cmake
    ${PATH_TO_LLVM}/lib/cmake/llvm
    ${PATH_TO_LLVM}/lib/cmake/clang
    ${PATH_TO_LLVM}/share/clang/cmake/
    ${PATH_TO_LLVM}/share/llvm/cmake/
  )

  message(STATUS "Searching in: ${search_paths}")
endif()


find_package(Clang
  REQUIRED CONFIG
  PATHS ${search_paths}
  )

find_package(LLVM
  ${SG_LLVM_VERSION}
  REQUIRED CONFIG
  )

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${CLANG_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})

add_executable(lsif-clang
    LsifClangMain.cpp
    # src/LsifSerialization.cpp
    # src/LsifMerge.cpp
    # src/LsifIndexAction.cpp
    # ${BACKWARD_ENABLE}
    )

# Make other directories available to us in clang-tools-extra
include_directories("../clangd/")
# add_subdirectory("../clangd/indexer/")


add_library(clangDaemonMinimal

  ../clangd/AST.cpp
  ../clangd/CodeCompletionStrings.cpp
  ../clangd/ExpectedTypes.cpp
  ../clangd/FindTarget.cpp
  ../clangd/FuzzyMatch.cpp
  ../clangd/Headers.cpp
  ../clangd/Protocol.cpp
  ../clangd/SourceCode.cpp
  ../clangd/URI.cpp
  ../clangd/support/Context.cpp
  ../clangd/support/Logger.cpp
  ../clangd/support/Trace.cpp

  # Cancellation.cpp
  # ClangdLSPServer.cpp
  # ClangdServer.cpp
  # CodeComplete.cpp
  # CompileCommands.cpp
  # Compiler.cpp
  # Diagnostics.cpp
  # DraftStore.cpp
  # FS.cpp
  # FSProvider.cpp
  # FileDistance.cpp
  # FindSymbols.cpp
  # Format.cpp
  # FormattedString.cpp
  # GlobalCompilationDatabase.cpp
  # HeaderSourceSwitch.cpp
  # Hover.cpp
  # IncludeFixer.cpp
  # JSONTransport.cpp
  # ParsedAST.cpp
  # PathMapping.cpp
  # Preamble.cpp
  # Quality.cpp
  # QueryDriverDatabase.cpp
  # RIFF.cpp
  # Selection.cpp
  # SemanticHighlighting.cpp
  # SemanticSelection.cpp
  # Shutdown.cpp
  # TUScheduler.cpp
  # Threading.cpp
  # XRefs.cpp

  # index/Background.cpp
  # index/BackgroundIndexLoader.cpp
  # index/BackgroundIndexStorage.cpp
  # index/BackgroundQueue.cpp
  # index/BackgroundRebuild.cpp
  # index/FileIndex.cpp
  # index/MemIndex.cpp
  # index/Serialization.cpp
  # index/YAMLSerialization.cpp
  ../clangd/index/CanonicalIncludes.cpp
  ../clangd/index/Index.cpp
  ../clangd/index/IndexAction.cpp
  ../clangd/index/Merge.cpp
  ../clangd/index/Ref.cpp
  ../clangd/index/Relation.cpp
  ../clangd/index/Symbol.cpp
  ../clangd/index/SymbolCollector.cpp
  ../clangd/index/SymbolID.cpp
  ../clangd/index/SymbolLocation.cpp
  ../clangd/index/SymbolOrigin.cpp

  # index/dex/Dex.cpp
  # index/dex/Iterator.cpp
  # index/dex/PostingList.cpp
  # index/dex/Trigram.cpp

  # refactor/Rename.cpp
  # refactor/Tweak.cpp
  )

set(CLANGD_ATOMIC_LIB "")
if(NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB)
  list(APPEND CLANGD_ATOMIC_LIB "atomic")
endif()

set(CLANG_DAEMON_LIBS
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
  # clangTidy
  clangTooling
  clangToolingCore
  clangToolingInclusions
  clangToolingRefactoring
  clangToolingSyntax
  ${LLVM_PTHREAD_LIB}
  # ${CLANGD_ATOMIC_LIB}
  ${ALL_CLANG_TIDY_CHECKS}
  m
)
  

if (LLVM IN_LIST LLVM_AVAILABLE_LIBS)
  target_link_libraries(clangDaemonMinimal
    LLVM
    ${CLANG_DAEMON_LIBS}
  )
else()
  target_link_libraries(clangDaemonMinimal
    LLVMSupport
    ${CLANG_DAEMON_LIBS}
  )
endif()

target_link_libraries(lsif-clang clangDaemonMinimal)

