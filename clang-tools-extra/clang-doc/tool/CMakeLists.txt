include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_executable(clang-doc
  ClangDocMain.cpp
  )

target_link_libraries(clang-doc
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangDoc
  clangTooling
  clangToolingCore
  )
  