set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(StaticAnalysisTests
  AnalyzerOptionsTest.cpp
  CallDescriptionTest.cpp
  CallEventTest.cpp
  StoreTest.cpp
  RegisterCustomCheckersTest.cpp
  SymbolReaperTest.cpp
  )

clang_target_link_libraries(StaticAnalysisTests
  PRIVATE
  clangBasic
  clangAnalysis
  clangAST
  clangASTMatchers
  clangCrossTU
  clangFrontend
  clangSerialization
  clangStaticAnalyzerCore
  clangStaticAnalyzerFrontend
  clangTooling
  )
