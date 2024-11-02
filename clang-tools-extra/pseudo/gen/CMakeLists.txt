set(LLVM_LINK_COMPONENTS Support)

add_clang_executable(pseudo-gen
  Main.cpp
  )

target_link_libraries(pseudo-gen
  PRIVATE
  clangPseudoGrammar
  )
