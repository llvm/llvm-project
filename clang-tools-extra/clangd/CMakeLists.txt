add_clang_executable(clangd
  ASTManager.cpp
  ClangDMain.cpp
  JSONRPCDispatcher.cpp
  Protocol.cpp
  ProtocolHandlers.cpp
  )

target_link_libraries(clangd
  clangBasic
  clangFormat
  clangFrontend
  clangTooling
  LLVMSupport
  )
