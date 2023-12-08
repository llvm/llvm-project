set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangPseudoGrammar
  Grammar.cpp
  GrammarBNF.cpp
  LRGraph.cpp
  LRTable.cpp
  LRTableBuild.cpp
  )

