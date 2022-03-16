set(LLVM_LINK_COMPONENTS support)

add_clang_tool(clang-pseudo
  ClangPseudo.cpp
  )

set(CLANG_PSEUDO_LIB_DEPS
  clangBasic
  clangToolingSyntaxPseudo
  )

clang_target_link_libraries(clang-pseudo
  PRIVATE
  ${CLANG_PSEUDO_LIB_DEPS}
  )
