include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(clang-read-diagnostics
  ClangReadDiagnostics.cpp
  )

clang_target_link_libraries(clang-read-diagnostics
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangSerialization
  clangTooling
  )
