set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkAnalyses
  CallGraph/CallGraphExtractor.cpp
  UnsafeBufferUsage/UnsafeBufferUsageExtractor.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangScalableStaticAnalysisFrameworkCore

  DEPENDS
  )
