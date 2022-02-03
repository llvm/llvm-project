set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(ClangPseudoTests
  GrammarTests.cpp
)

clang_target_link_libraries(ClangPseudoTests
  PRIVATE
  clangBasic
  clangLex
  clangSyntaxPseudo
  clangTesting
  )

target_link_libraries(ClangPseudoTests
  PRIVATE
  LLVMTestingSupport
  )
