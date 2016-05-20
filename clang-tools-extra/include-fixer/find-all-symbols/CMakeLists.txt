set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(findAllSymbols
  FindAllSymbols.cpp
  FindAllMacros.cpp
  PragmaCommentHandler.cpp
  SymbolInfo.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )

add_subdirectory(tool)
