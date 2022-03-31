set(LLVM_LINK_COMPONENTS
  Support
  )

add_custom_target(ClangPseudoUnitTests)
add_unittest(ClangPseudoUnitTests ClangPseudoTests
  DirectiveMapTest.cpp
  ForestTest.cpp
  GrammarTest.cpp
  LRTableTest.cpp
  TokenTest.cpp
)

clang_target_link_libraries(ClangPseudoTests
  PRIVATE
  clangBasic
  clangLex
  clangTesting
  )

target_link_libraries(ClangPseudoTests
  PRIVATE
  clangPseudo
  )
