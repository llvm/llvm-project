set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_clang_library(clangScalableStaticAnalysisTool
  Utils.cpp

  LINK_LIBS
  clangBasic
  clangScalableStaticAnalysisCore
  )
