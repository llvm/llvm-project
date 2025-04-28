set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(ClangAnalysisTests
  CFGDominatorTree.cpp
  CFGTest.cpp
  CloneDetectionTest.cpp
  ExprMutationAnalyzerTest.cpp
  IntervalPartitionTest.cpp
  MacroExpansionContextTest.cpp
  UnsafeBufferUsageTest.cpp
  )

clang_target_link_libraries(ClangAnalysisTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangAnalysis
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
