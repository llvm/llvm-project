set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(ParseTests
  ParseHLSLRootSignatureTest.cpp
  )

clang_target_link_libraries(ParseTests
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

target_link_libraries(ParseTests
  PRIVATE
  LLVMTestingAnnotations
  LLVMTestingSupport
  clangTesting
  )
