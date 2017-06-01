set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(ClangAnalysisTests
  CFGTest.cpp
  CloneDetectionTest.cpp
  )

target_link_libraries(ClangAnalysisTests
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  )
