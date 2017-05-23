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
  )

target_link_libraries(clangDaemon
  clangBasic
  clangFormat
  clangFrontend
  clangSema
  clangTooling
  clangToolingCore
  LLVMSupport
  )

add_subdirectory(tool)
