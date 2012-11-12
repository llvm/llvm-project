set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_unittest(ASTMatchersTests
  ASTMatchersTest.cpp)

target_link_libraries(ASTMatchersTests
  gtest gtest_main clangASTMatchers clangTooling)
