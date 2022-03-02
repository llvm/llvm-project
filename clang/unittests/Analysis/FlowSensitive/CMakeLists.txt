set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(ClangAnalysisFlowSensitiveTests
  DataflowAnalysisContextTest.cpp
  DataflowEnvironmentTest.cpp
  MapLatticeTest.cpp
  MultiVarConstantPropagationTest.cpp
  SingleVarConstantPropagationTest.cpp
  TestingSupport.cpp
  TestingSupportTest.cpp
  TransferTest.cpp
  TypeErasedDataflowAnalysisTest.cpp
  SolverTest.cpp
  )

clang_target_link_libraries(ClangAnalysisFlowSensitiveTests
  PRIVATE
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTesting
  clangTooling
  )

target_link_libraries(ClangAnalysisFlowSensitiveTests
  PRIVATE
  LLVMTestingSupport
  )
