set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangDaemon
  ClangdLSPServer.cpp
  ClangdServer.cpp
  ClangdUnit.cpp
  ClangdUnitStore.cpp
  DraftStore.cpp
  GlobalCompilationDatabase.cpp
  JSONRPCDispatcher.cpp
  JSONExpr.cpp
  Logger.cpp
  Protocol.cpp
  ProtocolHandlers.cpp
  Trace.cpp

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
  ${LLVM_PTHREAD_LIB}
  )

if( LLVM_LIB_FUZZING_ENGINE OR LLVM_USE_SANITIZE_COVERAGE )
  add_subdirectory(fuzzer)
endif()
add_subdirectory(tool)
