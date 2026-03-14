include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(clang-move
  ClangMove.cpp
  )

clang_target_link_libraries(clang-move
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  )
target_link_libraries(clang-move
  PRIVATE
  clangMove
  )
