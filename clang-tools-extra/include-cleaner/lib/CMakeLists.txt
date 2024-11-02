set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangIncludeCleaner
  Analysis.cpp
  HTMLReport.cpp
  Record.cpp
  Types.cpp
  WalkAST.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangToolingInclusionsStdlib
  )

