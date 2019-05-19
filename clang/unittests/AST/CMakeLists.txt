set(LLVM_LINK_COMPONENTS
  Support
  )

if (MSVC)
  set_source_files_properties(ASTImporterTest.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_clang_unittest(ASTTests
  ASTContextParentMapTest.cpp
  ASTImporterFixtures.cpp
  ASTImporterTest.cpp
  ASTImporterGenericRedeclTest.cpp
  ASTImporterVisibilityTest.cpp
  ASTTraverserTest.cpp
  ASTTypeTraitsTest.cpp
  ASTVectorTest.cpp
  CommentLexer.cpp
  CommentParser.cpp
  CommentTextTest.cpp
  DataCollectionTest.cpp
  DeclPrinterTest.cpp
  DeclTest.cpp
  EvaluateAsRValueTest.cpp
  ExternalASTSourceTest.cpp
  Language.cpp
  NamedDeclPrinterTest.cpp
  OMPStructuredBlockTest.cpp
  SourceLocationTest.cpp
  StmtPrinterTest.cpp
  StructuralEquivalenceTest.cpp
  )

target_link_libraries(ASTTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )
