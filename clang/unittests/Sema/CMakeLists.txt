# SemaTests are distinct to enable fast incremental builds.
# Merging it into AllClangTests would result in one less
# large statically linked binary, but separating it out is
# the right tradeoff today.
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
