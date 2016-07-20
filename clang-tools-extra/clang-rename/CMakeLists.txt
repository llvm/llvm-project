set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangRename
  USRFinder.cpp
  USRFindingAction.cpp
  USRLocFinder.cpp
  RenamingAction.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangIndex
  clangLex
  clangToolingCore
  )

add_subdirectory(tool)
