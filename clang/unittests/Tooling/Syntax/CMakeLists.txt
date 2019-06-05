set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(SyntaxTests
  TokensTest.cpp
)

target_link_libraries(SyntaxTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  clangToolingSyntax
  LLVMTestingSupport
  )
