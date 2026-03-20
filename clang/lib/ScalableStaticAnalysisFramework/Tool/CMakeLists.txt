set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangScalableStaticAnalysisFrameworkTool
  Utils.cpp

  LINK_LIBS
  clangBasic
  clangScalableStaticAnalysisFrameworkCore
  )
