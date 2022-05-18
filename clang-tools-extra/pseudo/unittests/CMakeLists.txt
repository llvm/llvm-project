set(LLVM_LINK_COMPONENTS
  Support
  TestingSupport
  )

add_custom_target(ClangPseudoUnitTests)
add_unittest(ClangPseudoUnitTests ClangPseudoTests
  BracketTest.cpp
  DirectiveTreeTest.cpp
  ForestTest.cpp
  GLRTest.cpp
  GrammarTest.cpp
  LRTableTest.cpp
  TokenTest.cpp
)

clang_target_link_libraries(ClangPseudoTests
  PRIVATE
  clangBasic
  clangLex
  )

target_link_libraries(ClangPseudoTests
  PRIVATE
  clangPseudo
  )
