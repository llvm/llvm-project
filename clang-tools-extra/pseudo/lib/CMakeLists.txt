set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangPseudo
  Bracket.cpp
  DirectiveTree.cpp
  Forest.cpp
  GLR.cpp
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
