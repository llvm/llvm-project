add_clang_unittest(SyntaxTests
  TreeTestBase.cpp
  BuildTreeTest.cpp
  MutationsTest.cpp
  SynthesisTest.cpp
  TreeTest.cpp
  TokensTest.cpp

  CLANG_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  clangToolingCore
  clangToolingSyntax

  LINK_LIBS
  clangTesting
  LLVMTestingAnnotations
  LLVMTestingSupport

  LLVM_COMPONENTS
  Support
)
