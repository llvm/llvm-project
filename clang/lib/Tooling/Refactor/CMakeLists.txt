set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingRefactor
  ASTSlice.cpp
  ASTStateSerialization.cpp
  Extract.cpp
  ExtractRepeatedExpressionIntoVariable.cpp
  ExtractionUtils.cpp
  FillInEnumSwitchCases.cpp
  FillInMissingMethodStubsFromAbstractClasses.cpp
  FillInMissingProtocolStubs.cpp
  IfSwitchConversion.cpp
  ImplementDeclaredMethods.cpp
  IndexerQueries.cpp
  LocalizeObjCStringLiteral.cpp
  RefactoringActions.cpp
  RefactoringActionFinder.cpp
  RefactoringOperation.cpp
  RefactoringOptions.cpp
  RenamingOperation.cpp
  RenameIndexedFile.cpp
  RenamedSymbol.cpp
  SourceLocationUtilities.cpp
  StmtUtils.cpp
  SymbolOperation.cpp
  SymbolOccurrenceFinder.cpp
  SymbolName.cpp
  SymbolUSRFinder.cpp
  TypeUtils.cpp
  USRFinder.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangEdit
  clangFrontend
  clangIndex
  clangLex
  clangToolingCore
  clangRewrite
  )
