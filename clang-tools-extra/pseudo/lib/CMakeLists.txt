set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangPseudo
  Bracket.cpp
  DirectiveTree.cpp
  Lex.cpp
  Token.cpp

  LINK_LIBS
  clangBasic
  clangLex

  DEPENDS
  ClangDriverOptions
  )

  target_include_directories(clangPseudo INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )
