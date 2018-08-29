set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyAbseilModule
  AbseilTidyModule.cpp
  DurationDivisionCheck.cpp
  FasterStrsplitDelimiterCheck.cpp
  NoNamespaceCheck.cpp
  StringFindStartswithCheck.cpp
  StrCatAppendCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
