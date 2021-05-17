set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )


add_clang_unittest(ASTTests
  ASTContextParentMapTest.cpp
  ASTImporterFixtures.cpp
  ASTImporterTest.cpp
  ASTImporterObjCTest.cpp
  ASTImporterGenericRedeclTest.cpp
  ASTImporterODRStrategiesTest.cpp
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
  NamedDeclPrinterTest.cpp
  RecursiveASTVisitorTest.cpp
  SizelessTypesTest.cpp
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
  clangTesting
  clangTooling
  )

target_link_libraries(ASTTests
  PRIVATE
  LLVMTestingSupport
)
