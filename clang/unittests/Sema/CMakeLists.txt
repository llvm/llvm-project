# There are only a small number of tests here,
# allow running them without building all of
# AllClangUnitTests.
add_distinct_clang_unittest(SemaTests
  ExternalSemaSourceTest.cpp
  CodeCompleteTest.cpp
  HeuristicResolverTest.cpp
  GslOwnerPointerInference.cpp
  SemaLookupTest.cpp
  SemaNoloadLookupTest.cpp
  CLANG_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangParse
  clangSema
  clangSerialization
  clangTooling
  LINK_LIBS
  LLVMTestingAnnotations
  LLVMTestingSupport
  clangTesting
  LLVM_COMPONENTS
  FrontendOpenMP
  Support
  )
