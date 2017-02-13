add_clang_executable(clangd
  ClangDMain.cpp
  JSONRPCDispatcher.cpp
  Protocol.cpp
  ProtocolHandlers.cpp
  )

target_link_libraries(clangd
  clangBasic
  clangFormat
  LLVMSupport
  )
