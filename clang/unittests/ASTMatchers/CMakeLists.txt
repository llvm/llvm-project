set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(ASTMatchersTests
  ASTMatchersInternalTest.cpp
  ASTMatchersNodeTest.cpp
  ASTMatchersNarrowingTest.cpp
  ASTMatchersTraversalTest.cpp
  GtestMatchersTest.cpp
  )

clang_target_link_libraries(ASTMatchersTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )

target_link_libraries(ASTMatchersTests
  PRIVATE
  clangTesting
  LLVMTestingSupport
)

add_subdirectory(Dynamic)
