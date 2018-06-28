set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(StaticAnalysisTests
  AnalyzerOptionsTest.cpp
  RegisterCustomCheckersTest.cpp
  )

target_link_libraries(StaticAnalysisTests
  PRIVATE
  clangBasic
  clangAnalysis
  clangFrontend
  clangStaticAnalyzerCore
  clangStaticAnalyzerFrontend
  clangTooling
  )
