set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangReorderFields
  ReorderFieldsAction.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangIndex
  clangLex
  clangToolingCore
  )

add_subdirectory(tool)
