include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_executable(clang-move
  ClangMoveMain.cpp
  )

target_link_libraries(clang-move
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangMove
  clangRewrite
  clangTooling
  clangToolingCore
  )
