set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(LexTests
  DependencyDirectivesScannerTest.cpp
  HeaderMapTest.cpp
  HeaderSearchTest.cpp
  LexerTest.cpp
  ModuleDeclStateTest.cpp
  PPCallbacksTest.cpp
  PPConditionalDirectiveRecordTest.cpp
  PPDependencyDirectivesTest.cpp
  PPMemoryAllocationsTest.cpp
  )

clang_target_link_libraries(LexTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangParse
  clangSema
  )

target_link_libraries(LexTests
  PRIVATE
  LLVMTestingSupport
  )
