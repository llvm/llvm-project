set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyAbseilModule
  AbseilTidyModule.cpp
  DurationDivisionCheck.cpp
  FasterStrsplitDelimiterCheck.cpp
  NoInternalDependenciesCheck.cpp
  NoNamespaceCheck.cpp
  RedundantStrcatCallsCheck.cpp
  StrCatAppendCheck.cpp
  StringFindStartswithCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  )
