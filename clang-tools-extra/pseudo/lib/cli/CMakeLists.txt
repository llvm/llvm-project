set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangPseudoCLI
  CLI.cpp

  # FIXME export the headers from clangPseudoCXX instead
  DEPENDS
  cxx_gen

  LINK_LIBS
  clangPseudoGrammar
  clangPseudoCXX
  )
