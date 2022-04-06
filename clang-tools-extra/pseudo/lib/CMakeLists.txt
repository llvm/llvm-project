set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangPseudo
  DirectiveTree.cpp
  Forest.cpp
  Grammar.cpp
  GrammarBNF.cpp
  Lex.cpp
  LRGraph.cpp
  LRTable.cpp
  LRTableBuild.cpp
  Token.cpp

  LINK_LIBS
  clangBasic
  clangLex
  )
