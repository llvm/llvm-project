add_distinct_clang_unittest(ClangScalableAnalysisFrameworkTests
  ASTEntityMappingTest.cpp
  BuildNamespaceTest.cpp
  EntityNameTest.cpp

  CLANG_LIBS
  clangAnalysisScalable
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  LINK_LIBS
  clangTesting
  LLVMTestingSupport
  LLVM_COMPONENTS
  FrontendOpenMP
  Support
  )

add_custom_target(ClangScalableAnalysisTests
  DEPENDS
  ClangScalableAnalysisFrameworkTests
  )
