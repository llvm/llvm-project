set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_unittest(ToolingTests
  CommentHandlerTest.cpp
  CompilationDatabaseTest.cpp
  ToolingTest.cpp
  RecursiveASTVisitorTest.cpp
  RefactoringTest.cpp
  RewriterTest.cpp
  RefactoringCallbacksTest.cpp
  )

target_link_libraries(ToolingTests
  clangAST
  clangTooling
  clangRewriteCore
  )
