set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  TargetParser
  )

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ASTMatchersDocTests.cpp
  COMMAND
    ${CLANG_SOURCE_DIR}/utils/generate_ast_matcher_doc_tests.py ARGS
    --input-file
    ${CLANG_SOURCE_DIR}/include/clang/ASTMatchers/ASTMatchers.h
    --output-file
    ${CMAKE_CURRENT_BINARY_DIR}/ASTMatchersDocTests.cpp
  MAIN_DEPENDENCY
    ${CLANG_SOURCE_DIR}/include/clang/ASTMatchers/ASTMatchers.h
  DEPENDS
    ${CLANG_SOURCE_DIR}/utils/generate_ast_matcher_doc_tests.py ARGS
)

add_clang_unittest(ASTMatchersTests
  ASTMatchersInternalTest.cpp
  ASTMatchersNodeTest.cpp
  ASTMatchersNarrowingTest.cpp
  ASTMatchersTraversalTest.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/ASTMatchersDocTests.cpp
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
