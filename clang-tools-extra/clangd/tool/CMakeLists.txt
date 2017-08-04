include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(clangd
  ClangdMain.cpp
  )

install(TARGETS clangd RUNTIME DESTINATION bin)

target_link_libraries(clangd
  clangBasic
  clangDaemon
  clangFormat
  clangFrontend
  clangSema
  clangTooling
  clangToolingCore
  LLVMSupport
  )
