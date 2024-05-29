set(LLVM_LINK_COMPONENTS
  Support
  )

add_custom_target(ClangPseudoUnitTests)
set_target_properties(ClangPseudoUnitTests PROPERTIES FOLDER "Clang Tools Extra/Tests")
add_unittest(ClangPseudoUnitTests ClangPseudoTests
  BracketTest.cpp
  CXXTest.cpp
  DirectiveTreeTest.cpp
  DisambiguateTest.cpp
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
  clangPseudoCXX
  clangPseudoGrammar
  LLVMTestingAnnotations
  LLVMTestingSupport
  )
