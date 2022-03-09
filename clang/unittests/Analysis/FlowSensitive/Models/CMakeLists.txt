set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(ClangAnalysisFlowSensitiveModelsTests
  UncheckedOptionalAccessModelTest.cpp
  )

clang_target_link_libraries(ClangAnalysisFlowSensitiveModelsTests
  PRIVATE
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAnalysisFlowSensitiveModels
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTesting
  clangTooling
  )

target_link_libraries(ClangAnalysisFlowSensitiveModelsTests
  PRIVATE
  LLVMTestingSupport
  )
