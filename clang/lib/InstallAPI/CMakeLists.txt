set(LLVM_LINK_COMPONENTS
  Support
  TextAPI
  )

add_clang_library(clangInstallAPI
  Context.cpp

  LINK_LIBS
  clangAST
  clangBasic
  )
