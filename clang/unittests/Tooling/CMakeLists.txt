set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  )

# By default MSVC has a 2^16 limit on the number of sections in an object file,
# and this needs more than that.
if (MSVC)
  set_source_files_properties(RecursiveASTVisitorTestExprVisitor.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_clang_unittest(ToolingTests
  CommentHandlerTest.cpp
  CompilationDatabaseTest.cpp
  LookupTest.cpp
  ToolingTest.cpp
  RecursiveASTVisitorTest.cpp
  RecursiveASTVisitorTestCallVisitor.cpp
  RecursiveASTVisitorTestDeclVisitor.cpp
  RecursiveASTVisitorTestExprVisitor.cpp
  RecursiveASTVisitorTestTypeLocVisitor.cpp
  RefactoringTest.cpp
  RewriterTest.cpp
  RefactoringCallbacksTest.cpp
  ReplacementsYamlTest.cpp
  QualTypeNamesTest.cpp
  )

target_link_libraries(ToolingTests
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangRewrite
  clangTooling
  clangToolingCore
  )
