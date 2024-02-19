set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangIncludeCleaner
  Analysis.cpp
  IncludeSpeller.cpp
  FindHeaders.cpp
  HTMLReport.cpp
  LocateSymbol.cpp
  Record.cpp
  Types.cpp
  WalkAST.cpp

  DEPENDS
  ClangDriverOptions
  )

clang_target_link_libraries(clangIncludeCleaner
  PRIVATE
  clangAST
  clangBasic
  clangFormat
  clangLex
  clangToolingCore
  clangToolingInclusions
  clangToolingInclusionsStdlib
  )

