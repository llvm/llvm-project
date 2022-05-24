add_clang_library(clangPseudoCXX
  CXX.cpp

  DEPENDS
  cxx_gen

  LINK_LIBS
  clangPseudoGrammar
  )
