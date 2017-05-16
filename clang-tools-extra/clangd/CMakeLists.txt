add_clang_executable(clangd
  ClangdLSPServer.cpp
  ClangdMain.cpp
  ClangdServer.cpp
  ClangdUnit.cpp
  ClangdUnitStore.cpp
  DraftStore.cpp
  GlobalCompilationDatabase.cpp
  JSONRPCDispatcher.cpp
  Protocol.cpp
  ProtocolHandlers.cpp
  )

install(TARGETS clangd RUNTIME DESTINATION bin)

target_link_libraries(clangd
  clangBasic
  clangFormat
  clangFrontend
  clangSema
  clangTooling
  clangToolingCore
  LLVMSupport
  )
