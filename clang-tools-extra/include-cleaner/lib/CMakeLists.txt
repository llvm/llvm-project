set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangIncludeCleaner
  HTMLReport.cpp
  Record.cpp
  WalkAST.cpp

  LINK_LIBS
  clangBasic
  clangAST
  )

