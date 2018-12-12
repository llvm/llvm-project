set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(ClangAnalysisTests
  CFGTest.cpp
  CloneDetectionTest.cpp
  ExprMutationAnalyzerTest.cpp
  )

target_link_libraries(ClangAnalysisTests
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )
