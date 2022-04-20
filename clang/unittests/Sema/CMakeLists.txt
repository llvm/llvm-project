set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(SemaTests
  ExternalSemaSourceTest.cpp
  CodeCompleteTest.cpp
  GslOwnerPointerInference.cpp
  SemaLookupTest.cpp
  )

clang_target_link_libraries(SemaTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangParse
  clangSema
  clangSerialization
  clangTooling
  )

target_link_libraries(SemaTests
  PRIVATE
  clangTesting
  LLVMTestingSupport
)
