include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_executable(clang-query ClangQuery.cpp)
target_link_libraries(clang-query
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangQuery
  clangTooling
  )

install(TARGETS clang-query RUNTIME DESTINATION bin)
