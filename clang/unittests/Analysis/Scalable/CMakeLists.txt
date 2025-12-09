add_distinct_clang_unittest(ClangScalableAnalysisFrameworkTests
  ASTEntityMappingTest.cpp
  BuildNamespaceTest.cpp
  EntityIdTest.cpp
  EntityIdTableTest.cpp
  EntityNameTest.cpp

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
