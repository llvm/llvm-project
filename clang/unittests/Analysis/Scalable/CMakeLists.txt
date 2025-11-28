add_distinct_clang_unittest(ClangScalableAnalysisFrameworkTests
  BuildNamespaceTest.cpp
  EntityNameTest.cpp
  ASTEntityMappingTest.cpp

  CLANG_LIBS
  clangAnalysisScalable
  clangAST
  clangASTMatchers
  clangBasic
  clangSerialization
  clangTooling
  )

add_custom_target(ClangScalableAnalysisTests
  DEPENDS
  ClangScalableAnalysisFrameworkTests
  )
