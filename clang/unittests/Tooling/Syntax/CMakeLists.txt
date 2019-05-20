set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  )

add_clang_unittest(TokensTest
  TokensTest.cpp
)

target_link_libraries(TokensTest
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
