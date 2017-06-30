set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_library(clangToolingRefactor
  AtomicChange.cpp
  Rename/RenamingAction.cpp
  Rename/USRFinder.cpp
  Rename/USRFindingAction.cpp
  Rename/USRLocFinder.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangIndex
  clangLex
  clangToolingCore
  )
