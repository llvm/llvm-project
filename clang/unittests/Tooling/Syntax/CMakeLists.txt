set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(SyntaxTests
  TreeTestBase.cpp
  BuildTreeTest.cpp
  MutationsTest.cpp
  TokensTest.cpp
)

clang_target_link_libraries(SyntaxTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTesting
  clangTooling
  clangToolingCore
  clangToolingSyntax
  )

target_link_libraries(SyntaxTests
  PRIVATE
  LLVMTestingSupport
)
