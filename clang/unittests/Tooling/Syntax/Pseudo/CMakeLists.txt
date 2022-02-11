set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(ClangPseudoTests
  GrammarTest.cpp
  LRTableTest.cpp
)

clang_target_link_libraries(ClangPseudoTests
  PRIVATE
  clangBasic
  clangLex
  clangToolingSyntaxPseudo
  clangTesting
  )

target_link_libraries(ClangPseudoTests
  PRIVATE
  LLVMTestingSupport
  )
