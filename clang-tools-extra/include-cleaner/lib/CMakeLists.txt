set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangIncludeCleaner
  Analysis.cpp
  FindHeaders.cpp
  HTMLReport.cpp
  LocateSymbol.cpp
  Record.cpp
  Types.cpp
  WalkAST.cpp
  )

clang_target_link_libraries(clangIncludeCleaner
  PRIVATE
  clangAST
  clangBasic
  clangLex
  clangToolingInclusionsStdlib
  )

