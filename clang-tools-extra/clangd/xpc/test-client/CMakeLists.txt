add_clang_tool(
  clangd-xpc-test-client
  ClangdXPCTestClient.cpp

  DEPENDS ClangdXPC
)

set(LLVM_LINK_COMPONENTS
    support
)

clang_target_link_libraries(clangd-xpc-test-client
  PRIVATE
  clangBasic
  clangFormat
  clangFrontend
  clangSema
  clangTooling
  clangToolingCore
)
target_link_libraries(clangd-xpc-test-client
  PRIVATE
  clangDaemon
  clangdXpcJsonConversions
)
