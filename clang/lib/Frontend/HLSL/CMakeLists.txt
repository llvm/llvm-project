set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangHLSLFrontend
  FrontendActions.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangParse
  clangSema
  )
