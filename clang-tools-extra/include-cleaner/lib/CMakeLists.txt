set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangIncludeCleaner
  WalkAST.cpp

  LINK_LIBS
  clangBasic
  clangAST
  )

