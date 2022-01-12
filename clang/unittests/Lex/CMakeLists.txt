set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(LexTests
  DependencyDirectivesSourceMinimizerTest.cpp
  HeaderMapTest.cpp
  HeaderSearchTest.cpp
  LexerTest.cpp
  PPCallbacksTest.cpp
  PPConditionalDirectiveRecordTest.cpp
  )

clang_target_link_libraries(LexTests
  PRIVATE
  clangAST
  clangBasic
  clangLex
  clangParse
  clangSema
  )
