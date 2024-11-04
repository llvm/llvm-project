set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangPseudoCXX
  CXX.cpp

  DEPENDS
  cxx_gen

  LINK_LIBS
  clangPseudo
  clangPseudoGrammar
  )

clang_target_link_libraries(clangPseudoCXX
  PRIVATE
  clangBasic
  )
