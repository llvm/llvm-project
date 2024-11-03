set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(ClangAnalysisTests
  CFGDominatorTree.cpp
  CFGTest.cpp
  CloneDetectionTest.cpp
  ExprMutationAnalyzerTest.cpp
  MacroExpansionContextTest.cpp
  UnsafeBufferUsageTest.cpp
  )

clang_target_link_libraries(ClangAnalysisTests
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )

target_link_libraries(ClangAnalysisTests
  PRIVATE
  clangTesting
  LLVMTestingSupport
  )

add_subdirectory(FlowSensitive)
