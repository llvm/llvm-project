set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Core
  FrontendOpenMP
  Support
  TargetParser
  )

# FIXME: the entry points to the interpreter should be moved out of clangAST
# into the parser or Sema in order to allow the interpreter to be moved to
# another library which depends on clangAST.
clang_tablegen(Opcodes.inc
  -gen-clang-opcodes
  SOURCE ByteCode/Opcodes.td
  TARGET Opcodes)

clang_tablegen(AttrDocTable.inc -gen-clang-attr-doc-table
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../include/
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../include/clang/Basic/Attr.td
  TARGET ClangAttrDocTable)

add_clang_library(clangAST
  APValue.cpp
  ASTConcept.cpp
  ASTConsumer.cpp
  ASTContext.cpp
  ASTDiagnostic.cpp
  ASTDumper.cpp
  ASTImporter.cpp
  ASTImporterLookupTable.cpp
  ASTStructuralEquivalence.cpp
  ASTTypeTraits.cpp
  AttrDocTable.cpp
  AttrImpl.cpp
  Availability.cpp
  Comment.cpp
  CommentBriefParser.cpp
  CommentCommandTraits.cpp
  CommentLexer.cpp
  CommentParser.cpp
  CommentSema.cpp
  ComparisonCategories.cpp
  ComputeDependence.cpp
  CXXInheritance.cpp
  DataCollection.cpp
  Decl.cpp
  DeclarationName.cpp
  DeclBase.cpp
  DeclCXX.cpp
  DeclFriend.cpp
  DeclGroup.cpp
  DeclObjC.cpp
  DeclOpenMP.cpp
  DeclPrinter.cpp
  DeclTemplate.cpp
  DynamicRecursiveASTVisitor.cpp
  ParentMapContext.cpp
  Expr.cpp
  ExprClassification.cpp
  ExprConcepts.cpp
  ExprConstant.cpp
  ExprCXX.cpp
  ExprObjC.cpp
  ExternalASTMerger.cpp
  ExternalASTSource.cpp
  FormatString.cpp
  InheritViz.cpp
  ByteCode/BitcastBuffer.cpp
  ByteCode/ByteCodeEmitter.cpp
  ByteCode/Compiler.cpp
  ByteCode/Context.cpp
  ByteCode/Descriptor.cpp
  ByteCode/Disasm.cpp
  ByteCode/EvalEmitter.cpp
  ByteCode/Frame.cpp
  ByteCode/Function.cpp
  ByteCode/FunctionPointer.cpp
  ByteCode/InterpBuiltin.cpp
  ByteCode/InterpBuiltinBitCast.cpp
  ByteCode/Floating.cpp
  ByteCode/EvaluationResult.cpp
  ByteCode/DynamicAllocator.cpp
  ByteCode/Interp.cpp
  ByteCode/InterpBlock.cpp
  ByteCode/InterpFrame.cpp
  ByteCode/InterpStack.cpp
  ByteCode/InterpState.cpp
  ByteCode/Pointer.cpp
  ByteCode/PrimType.cpp
  ByteCode/Program.cpp
  ByteCode/Record.cpp
  ByteCode/Source.cpp
  ByteCode/State.cpp
  ByteCode/MemberPointer.cpp
  ByteCode/InterpShared.cpp
  ItaniumCXXABI.cpp
  ItaniumMangle.cpp
  JSONNodeDumper.cpp
  Mangle.cpp
  MicrosoftCXXABI.cpp
  MicrosoftMangle.cpp
  NestedNameSpecifier.cpp
  NSAPI.cpp
  ODRDiagsEmitter.cpp
  ODRHash.cpp
  OpenACCClause.cpp
  OpenMPClause.cpp
  OSLog.cpp
  ParentMap.cpp
  PrintfFormatString.cpp
  QualTypeNames.cpp
  Randstruct.cpp
  RawCommentList.cpp
  RecordLayout.cpp
  RecordLayoutBuilder.cpp
  ScanfFormatString.cpp
  SelectorLocationsKind.cpp
  Stmt.cpp
  StmtCXX.cpp
  StmtIterator.cpp
  StmtObjC.cpp
  StmtOpenACC.cpp
  StmtOpenMP.cpp
  StmtPrinter.cpp
  StmtProfile.cpp
  StmtViz.cpp
  TemplateBase.cpp
  TemplateName.cpp
  TextNodeDumper.cpp
  Type.cpp
  TypeLoc.cpp
  TypePrinter.cpp
  VTableBuilder.cpp
  VTTBuilder.cpp

  LINK_LIBS
  clangBasic
  clangLex

  DEPENDS
  ClangAttrDocTable
  Opcodes
  omp_gen
  ClangDriverOptions
  intrinsics_gen
  # These generated headers are included transitively.
  AArch64TargetParserTableGen
  )
