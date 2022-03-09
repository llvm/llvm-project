set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(ClangAnalysisFlowSensitiveTestingTests
  DataflowCheckerTest.cpp
  )

clang_target_link_libraries(ClangAnalysisFlowSensitiveTestingTests
  PRIVATE
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAnalysisFlowSensitiveTesting
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTesting
  clangTooling
  )

target_link_libraries(ClangAnalysisFlowSensitiveTestingTests
  PRIVATE
  LLVMTestingSupport
  )
