set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-query
  )

add_extra_unittest(ClangQueryTests
  QueryEngineTest.cpp
  QueryParserTest.cpp
  )

clang_target_link_libraries(ClangQueryTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangSerialization
  clangTooling
  )
target_link_libraries(ClangQueryTests
  PRIVATE
  clangQuery
  )
