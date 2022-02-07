set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangToolingSyntaxPseudo
  Grammar.cpp
  GrammarBNF.cpp
  Lex.cpp
  LRGraph.cpp
  LRTable.cpp
  LRTableBuild.cpp
  Preprocess.cpp
  Token.cpp

  LINK_LIBS
  clangBasic
  clangLex
  )
