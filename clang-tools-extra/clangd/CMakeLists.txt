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
  Protocol.cpp
  ProtocolHandlers.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFormat
  clangFrontend
  clangIndex
  clangLex
  clangSema
  clangTooling
  clangToolingCore
  ${LLVM_PTHREAD_LIB}
  )

add_subdirectory(tool)
