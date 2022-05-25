set(LLVM_LINK_COMPONENTS support)

add_clang_tool(clang-pseudo
  ClangPseudo.cpp
  )

clang_target_link_libraries(clang-pseudo
  PRIVATE
  clangBasic
  )

target_link_libraries(clang-pseudo
  PRIVATE
  clangPseudo
  clangPseudoGrammar
  )

