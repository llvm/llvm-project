add_clang_unittest(ASTTests
  CommentLexer.cpp
  CommentParser.cpp
  DeclPrinterTest.cpp
  SourceLocationTest.cpp
  StmtPrinterTest.cpp
  )

target_link_libraries(ASTTests
  clangAST clangASTMatchers clangTooling
  )
