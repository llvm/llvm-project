set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(ClangAnalysisFlowSensitiveTests
  SingleVarConstantPropagationTest.cpp
  TestingSupport.cpp
  TestingSupportTest.cpp
  TypeErasedDataflowAnalysisTest.cpp
  )

clang_target_link_libraries(ClangAnalysisFlowSensitiveTests
  PRIVATE
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTesting
  clangTooling
  )

target_link_libraries(ClangAnalysisFlowSensitiveTests
  PRIVATE
  LLVMTestingSupport
  )
