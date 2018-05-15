set(LLVM_LINK_COMPONENTS
  Support
  )

set(CLANGD_ATOMIC_LIB "")
if(NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB)
  list(APPEND CLANGD_ATOMIC_LIB "atomic")
endif()

add_clang_library(clangDaemon
  AST.cpp
  ClangdLSPServer.cpp
  ClangdServer.cpp
  ClangdUnit.cpp
  CodeComplete.cpp
  CodeCompletionStrings.cpp
  CompileArgsCache.cpp
  Compiler.cpp
  Context.cpp
  Diagnostics.cpp
  DraftStore.cpp
  FindSymbols.cpp
  FuzzyMatch.cpp
  GlobalCompilationDatabase.cpp
  Headers.cpp
  JSONExpr.cpp
  JSONRPCDispatcher.cpp
  Logger.cpp
  Protocol.cpp
  ProtocolHandlers.cpp
  Quality.cpp
  SourceCode.cpp
  Threading.cpp
  Trace.cpp
  TUScheduler.cpp
  URI.cpp
  XRefs.cpp
  index/CanonicalIncludes.cpp
  index/FileIndex.cpp
  index/Index.cpp
  index/MemIndex.cpp
  index/Merge.cpp
  index/SymbolCollector.cpp
  index/SymbolYAML.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangIndex
  clangLex
  clangSema
  clangSerialization
  clangTooling
  clangToolingCore
  clangToolingRefactor
  ${LLVM_PTHREAD_LIB}
  ${CLANGD_ATOMIC_LIB}
  )

if( LLVM_LIB_FUZZING_ENGINE OR LLVM_USE_SANITIZE_COVERAGE )
  add_subdirectory(fuzzer)
endif()
add_subdirectory(tool)
add_subdirectory(global-symbol-builder)
