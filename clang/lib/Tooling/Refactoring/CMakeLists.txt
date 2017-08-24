set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangToolingRefactor
  ASTSelection.cpp
  AtomicChange.cpp
  Rename/RenamingAction.cpp
  Rename/SymbolOccurrences.cpp
  Rename/USRFinder.cpp
  Rename/USRFindingAction.cpp
  Rename/USRLocFinder.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangIndex
  clangLex
  clangToolingCore
  )
