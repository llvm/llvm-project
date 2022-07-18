add_subdirectory(cli)
add_subdirectory(cxx)
add_subdirectory(grammar)

set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangPseudo
  Bracket.cpp
  DirectiveTree.cpp
  Forest.cpp
  GLR.cpp
  Lex.cpp
  Token.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangPseudoGrammar
  )

  target_include_directories(clangPseudo INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )
