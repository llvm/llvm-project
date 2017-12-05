set(LLVM_LINK_COMPONENTS
  support
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-query
  )

add_extra_unittest(ClangQueryTests
  QueryEngineTest.cpp
  QueryParserTest.cpp
  )

target_link_libraries(ClangQueryTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangQuery
  clangTooling
  )
