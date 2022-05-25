set(LLVM_LINK_COMPONENTS Support)

# This library intents to keep as minimal dependencies as possible, it is a base
# library of the cxx generator, to avoid creating long dep paths in the build
# graph.
add_clang_library(clangPseudoGrammar
  Grammar.cpp
  GrammarBNF.cpp
  LRGraph.cpp
  LRTable.cpp
  LRTableBuild.cpp

  # FIXME: can we get rid of the clangBasic dependency? We need it for the
  # clang::tok::getTokenName and clang::tok::getPunctuatorSpelling functions, we
  # could consider remimplement these functions.
  LINK_LIBS
  clangBasic
  )
