set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangToolingRefactoring
  ASTSelection.cpp
  ASTSelectionRequirements.cpp
  AtomicChange.cpp
  Extract/Extract.cpp
  Extract/SourceExtraction.cpp
  Lookup.cpp
  RefactoringActions.cpp
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
  clangLex
  clangRewrite
  clangToolingCore
  clangUnifiedSymbolResolution

  DEPENDS
  omp_gen
  )
