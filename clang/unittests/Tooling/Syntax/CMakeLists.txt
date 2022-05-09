set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(SyntaxTests
  TreeTestBase.cpp
  BuildTreeTest.cpp
  MutationsTest.cpp
  SynthesisTest.cpp
  TreeTest.cpp
  TokensTest.cpp
)

clang_target_link_libraries(SyntaxTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  clangToolingCore
  clangToolingSyntax
  )

target_link_libraries(SyntaxTests
  PRIVATE
  clangTesting
  LLVMTestingSupport
)
