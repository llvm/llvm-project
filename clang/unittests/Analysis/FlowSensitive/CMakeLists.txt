set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(ClangAnalysisFlowSensitiveTests
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
  clangTesting
  clangTooling
  )

