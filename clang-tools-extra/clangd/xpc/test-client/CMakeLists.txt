include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

add_clang_tool(
  clangd-xpc-test-client
  ClangdXPCTestClient.cpp

  DEPENDS ClangdXPC
)

set(LLVM_LINK_COMPONENTS
    support
)

target_link_libraries(clangd-xpc-test-client
  PRIVATE
  clangBasic
  clangDaemon
  clangFormat
  clangFrontend
  clangSema
  clangTooling
  clangToolingCore
  clangdXpcJsonConversions
)
