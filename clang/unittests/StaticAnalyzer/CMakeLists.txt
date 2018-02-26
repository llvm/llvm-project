set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(StaticAnalysisTests
  AnalyzerOptionsTest.cpp
  )

target_link_libraries(StaticAnalysisTests
  PRIVATE
  clangBasic
  clangAnalysis
  clangStaticAnalyzerCore 
  )
