set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangAnalysisFlowSensitiveNullability
  PointerNullability.cpp
  PointerNullabilityAnalysis.cpp
  PointerNullabilityMatchers.cpp
  ValueTransferer.cpp

  LINK_LIBS
  clangAnalysis
  clangAnalysisFlowSensitive
  clangAST
  clangASTMatchers
  clangBasic
  )
