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
  ASTImporterODRStrategiesTest.cpp
  ASTImporterVisibilityTest.cpp
  ASTTraverserTest.cpp
  ASTTypeTraitsTest.cpp
  ASTTraverserTest.cpp
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
  RecursiveASTVisitorTest.cpp
  SourceLocationTest.cpp
  StmtPrinterTest.cpp
  StructuralEquivalenceTest.cpp
  )

clang_target_link_libraries(ASTTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )
