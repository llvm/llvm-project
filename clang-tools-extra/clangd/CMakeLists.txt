add_clang_executable(clangd
  ASTManager.cpp
  ClangDMain.cpp
  JSONRPCDispatcher.cpp
  Protocol.cpp
  ProtocolHandlers.cpp
  )

install(TARGETS clangd RUNTIME DESTINATION bin)

target_link_libraries(clangd
  clangBasic
  clangFormat
  clangFrontend
  clangTooling
  clangToolingCore
  LLVMSupport
  )
