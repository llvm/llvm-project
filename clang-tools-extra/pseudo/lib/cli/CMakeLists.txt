set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangPseudoCLI
  CLI.cpp

  LINK_LIBS
  clangPseudoGrammar
  clangPseudoCXX
  )
