set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangIncludeCleaner STATIC
  Analysis.cpp
  FindHeaders.cpp
  HTMLReport.cpp
  IncludeSpeller.cpp
  LocateSymbol.cpp
  MappingFile.cpp
  Record.cpp
  Types.cpp
  WalkAST.cpp

  DEPENDS
  ClangDriverOptions

  # Workaround for Clang bug: https://github.com/llvm/llvm-project/issues/184559
  DISABLE_PCH_REUSE
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

