set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangDaemon
  ClangdLSPServer.cpp
  ClangdServer.cpp
  ClangdUnit.cpp
  ClangdUnitStore.cpp
  CodeComplete.cpp
  Context.cpp
  Compiler.cpp
  DraftStore.cpp
  FuzzyMatch.cpp
  GlobalCompilationDatabase.cpp
  JSONExpr.cpp
  JSONRPCDispatcher.cpp
  Logger.cpp
  Protocol.cpp
  ProtocolHandlers.cpp
  Trace.cpp
  index/MemIndex.cpp
  index/Index.cpp
  index/SymbolCollector.cpp
  index/SymbolYAML.cpp

  LINK_LIBS
  clangAST
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
  )

if( LLVM_LIB_FUZZING_ENGINE OR LLVM_USE_SANITIZE_COVERAGE )
  add_subdirectory(fuzzer)
endif()
add_subdirectory(tool)
