//===--- ASTImporter.cpp - Importing ASTs from other Contexts ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTImporter class which imports AST nodes from one
//  context into another context.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include <deque>

namespace clang {
  class ASTNodeImporter : public TypeVisitor<ASTNodeImporter, QualType>,
                          public DeclVisitor<ASTNodeImporter, Decl *>,
                          public StmtVisitor<ASTNodeImporter, Stmt *> {
    ASTImporter &Importer;

  public:
    explicit ASTNodeImporter(ASTImporter &Importer) : Importer(Importer) { }
    
    using TypeVisitor<ASTNodeImporter, QualType>::Visit;
    using DeclVisitor<ASTNodeImporter, Decl *>::Visit;
    using StmtVisitor<ASTNodeImporter, Stmt *>::Visit;

    // Importing types
    QualType VisitType(const Type *T);
    QualType VisitAtomicType(const AtomicType *T);
    QualType VisitBuiltinType(const BuiltinType *T);
    QualType VisitDecayedType(const DecayedType *T);
    QualType VisitComplexType(const ComplexType *T);
    QualType VisitPointerType(const PointerType *T);
    QualType VisitBlockPointerType(const BlockPointerType *T);
    QualType VisitLValueReferenceType(const LValueReferenceType *T);
    QualType VisitRValueReferenceType(const RValueReferenceType *T);
    QualType VisitMemberPointerType(const MemberPointerType *T);
    QualType VisitConstantArrayType(const ConstantArrayType *T);
    QualType VisitIncompleteArrayType(const IncompleteArrayType *T);
    QualType VisitVariableArrayType(const VariableArrayType *T);
    // FIXME: DependentSizedArrayType
    // FIXME: DependentSizedExtVectorType
    QualType VisitVectorType(const VectorType *T);
    QualType VisitExtVectorType(const ExtVectorType *T);
    QualType VisitFunctionNoProtoType(const FunctionNoProtoType *T);
    QualType VisitFunctionProtoType(const FunctionProtoType *T);
    // FIXME: UnresolvedUsingType
    QualType VisitParenType(const ParenType *T);
    QualType VisitTypedefType(const TypedefType *T);
    QualType VisitTypeOfExprType(const TypeOfExprType *T);
    // FIXME: DependentTypeOfExprType
    QualType VisitTypeOfType(const TypeOfType *T);
    QualType VisitDecltypeType(const DecltypeType *T);
    QualType VisitUnaryTransformType(const UnaryTransformType *T);
    QualType VisitAutoType(const AutoType *T);
    QualType VisitInjectedClassNameType(const InjectedClassNameType *T);
    // FIXME: DependentDecltypeType
    QualType VisitRecordType(const RecordType *T);
    QualType VisitEnumType(const EnumType *T);
    QualType VisitAttributedType(const AttributedType *T);
    QualType VisitTemplateTypeParmType(const TemplateTypeParmType *T);
    QualType VisitSubstTemplateTypeParmType(const SubstTemplateTypeParmType *T);
    QualType VisitTemplateSpecializationType(const TemplateSpecializationType *T);
    QualType VisitElaboratedType(const ElaboratedType *T);
    // FIXME: DependentNameType
    // FIXME: DependentTemplateSpecializationType
    QualType VisitObjCInterfaceType(const ObjCInterfaceType *T);
    QualType VisitObjCObjectType(const ObjCObjectType *T);
    QualType VisitObjCObjectPointerType(const ObjCObjectPointerType *T);
                            
    // Importing declarations                            
    bool ImportDeclParts(NamedDecl *D, DeclContext *&DC, 
                         DeclContext *&LexicalDC, DeclarationName &Name, 
                         NamedDecl *&ToD, SourceLocation &Loc);
    void ImportDefinitionIfNeeded(Decl *FromD, Decl *ToD = nullptr);
    void ImportDeclarationNameLoc(const DeclarationNameInfo &From,
                                  DeclarationNameInfo& To);
    void ImportDeclContext(DeclContext *FromDC, bool ForceImport = false);

    bool ImportCastPath(CastExpr *E, CXXCastPath &Path);

    typedef DesignatedInitExpr::Designator Designator;
    Designator ImportDesignator(const Designator &D);

                        
    /// \brief What we should import from the definition.
    enum ImportDefinitionKind { 
      /// \brief Import the default subset of the definition, which might be
      /// nothing (if minimal import is set) or might be everything (if minimal
      /// import is not set).
      IDK_Default,
      /// \brief Import everything.
      IDK_Everything,
      /// \brief Import only the bare bones needed to establish a valid
      /// DeclContext.
      IDK_Basic
    };

    bool shouldForceImportDeclContext(ImportDefinitionKind IDK) {
      return IDK == IDK_Everything ||
             (IDK == IDK_Default && !Importer.isMinimalImport());
    }

    bool ImportDefinition(RecordDecl *From, RecordDecl *To, 
                          ImportDefinitionKind Kind = IDK_Default);
    bool ImportDefinition(VarDecl *From, VarDecl *To,
                          ImportDefinitionKind Kind = IDK_Default);
    bool ImportDefinition(EnumDecl *From, EnumDecl *To,
                          ImportDefinitionKind Kind = IDK_Default);
    bool ImportDefinition(ObjCInterfaceDecl *From, ObjCInterfaceDecl *To,
                          ImportDefinitionKind Kind = IDK_Default);
    bool ImportDefinition(ObjCProtocolDecl *From, ObjCProtocolDecl *To,
                          ImportDefinitionKind Kind = IDK_Default);
    TemplateParameterList *ImportTemplateParameterList(
                                                 TemplateParameterList *Params);
    TemplateArgument ImportTemplateArgument(const TemplateArgument &From);
    TemplateArgumentLoc ImportTemplateArgumentLoc(
        const TemplateArgumentLoc &TALoc, bool &Error);
    bool ImportTemplateArguments(const TemplateArgument *FromArgs,
                                 unsigned NumFromArgs,
                               SmallVectorImpl<TemplateArgument> &ToArgs);
    bool IsStructuralMatch(RecordDecl *FromRecord, RecordDecl *ToRecord,
                           bool Complain = true);
    bool IsStructuralMatch(VarDecl *FromVar, VarDecl *ToVar,
                           bool Complain = true);
    bool IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToRecord);
    bool IsStructuralMatch(EnumConstantDecl *FromEC, EnumConstantDecl *ToEC);
    bool IsStructuralMatch(ClassTemplateDecl *From, ClassTemplateDecl *To);
    bool IsStructuralMatch(VarTemplateDecl *From, VarTemplateDecl *To);
    Decl *VisitDecl(Decl *D);
    Decl *VisitAccessSpecDecl(AccessSpecDecl *D);
    Decl *VisitStaticAssertDecl(StaticAssertDecl *D);
    Decl *VisitTranslationUnitDecl(TranslationUnitDecl *D);
    Decl *VisitNamespaceDecl(NamespaceDecl *D);
    Decl *VisitTypedefNameDecl(TypedefNameDecl *D, bool IsAlias);
    Decl *VisitTypedefDecl(TypedefDecl *D);
    Decl *VisitTypeAliasDecl(TypeAliasDecl *D);
    Decl *VisitLabelDecl(LabelDecl *D);
    Decl *VisitEnumDecl(EnumDecl *D);
    Decl *VisitRecordDecl(RecordDecl *D);
    Decl *VisitEnumConstantDecl(EnumConstantDecl *D);
    Decl *VisitFunctionDecl(FunctionDecl *D);
    Decl *VisitCXXMethodDecl(CXXMethodDecl *D);
    Decl *VisitCXXConstructorDecl(CXXConstructorDecl *D);
    Decl *VisitCXXDestructorDecl(CXXDestructorDecl *D);
    Decl *VisitCXXConversionDecl(CXXConversionDecl *D);
    Decl *VisitFieldDecl(FieldDecl *D);
    Decl *VisitIndirectFieldDecl(IndirectFieldDecl *D);
    Decl *VisitFriendDecl(FriendDecl *D);
    Decl *VisitObjCIvarDecl(ObjCIvarDecl *D);
    Decl *VisitVarDecl(VarDecl *D);
    Decl *VisitImplicitParamDecl(ImplicitParamDecl *D);
    Decl *VisitParmVarDecl(ParmVarDecl *D);
    Decl *VisitObjCMethodDecl(ObjCMethodDecl *D);
    Decl *VisitObjCTypeParamDecl(ObjCTypeParamDecl *D);
    Decl *VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    Decl *VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    Decl *VisitLinkageSpecDecl(LinkageSpecDecl *D);

    ObjCTypeParamList *ImportObjCTypeParamList(ObjCTypeParamList *list);
    Decl *VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    Decl *VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    Decl *VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    Decl *VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    Decl *VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    Decl *VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    Decl *VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    Decl *VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    Decl *VisitClassTemplateDecl(ClassTemplateDecl *D);
    Decl *VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    Decl *VisitVarTemplateDecl(VarTemplateDecl *D);
    Decl *VisitVarTemplateSpecializationDecl(VarTemplateSpecializationDecl *D);

    // Importing statements
    DeclGroupRef ImportDeclGroup(DeclGroupRef DG);

    Stmt *VisitStmt(Stmt *S);
    Stmt *VisitGCCAsmStmt(GCCAsmStmt *S);
    Stmt *VisitDeclStmt(DeclStmt *S);
    Stmt *VisitNullStmt(NullStmt *S);
    Stmt *VisitCompoundStmt(CompoundStmt *S);
    Stmt *VisitCaseStmt(CaseStmt *S);
    Stmt *VisitDefaultStmt(DefaultStmt *S);
    Stmt *VisitLabelStmt(LabelStmt *S);
    Stmt *VisitAttributedStmt(AttributedStmt *S);
    Stmt *VisitIfStmt(IfStmt *S);
    Stmt *VisitSwitchStmt(SwitchStmt *S);
    Stmt *VisitWhileStmt(WhileStmt *S);
    Stmt *VisitDoStmt(DoStmt *S);
    Stmt *VisitForStmt(ForStmt *S);
    Stmt *VisitGotoStmt(GotoStmt *S);
    Stmt *VisitIndirectGotoStmt(IndirectGotoStmt *S);
    Stmt *VisitContinueStmt(ContinueStmt *S);
    Stmt *VisitBreakStmt(BreakStmt *S);
    Stmt *VisitReturnStmt(ReturnStmt *S);
    // FIXME: MSAsmStmt
    // FIXME: SEHExceptStmt
    // FIXME: SEHFinallyStmt
    // FIXME: SEHTryStmt
    // FIXME: SEHLeaveStmt
    // FIXME: CapturedStmt
    Stmt *VisitCXXCatchStmt(CXXCatchStmt *S);
    Stmt *VisitCXXTryStmt(CXXTryStmt *S);
    Stmt *VisitCXXForRangeStmt(CXXForRangeStmt *S);
    // FIXME: MSDependentExistsStmt
    Stmt *VisitObjCForCollectionStmt(ObjCForCollectionStmt *S);
    Stmt *VisitObjCAtCatchStmt(ObjCAtCatchStmt *S);
    Stmt *VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S);
    Stmt *VisitObjCAtTryStmt(ObjCAtTryStmt *S);
    Stmt *VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S);
    Stmt *VisitObjCAtThrowStmt(ObjCAtThrowStmt *S);
    Stmt *VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S);

    // Importing expressions
    Expr *VisitExpr(Expr *E);
    Expr *VisitVAArgExpr(VAArgExpr *E);
    Expr *VisitGNUNullExpr(GNUNullExpr *E);
    Expr *VisitPredefinedExpr(PredefinedExpr *E);
    Expr *VisitDeclRefExpr(DeclRefExpr *E);
    Expr *VisitImplicitValueInitExpr(ImplicitValueInitExpr *ILE);
    Expr *VisitDesignatedInitExpr(DesignatedInitExpr *E);
    Expr *VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E);
    Expr *VisitIntegerLiteral(IntegerLiteral *E);
    Expr *VisitFloatingLiteral(FloatingLiteral *E);
    Expr *VisitCharacterLiteral(CharacterLiteral *E);
    Expr *VisitStringLiteral(StringLiteral *E);
    Expr *VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    Expr *VisitAtomicExpr(AtomicExpr *E);
    Expr *VisitAddrLabelExpr(AddrLabelExpr *E);
    Expr *VisitParenExpr(ParenExpr *E);
    Expr *VisitParenListExpr(ParenListExpr *E);
    Expr *VisitStmtExpr(StmtExpr *E);
    Expr *VisitUnaryOperator(UnaryOperator *E);
    Expr *VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
    Expr *VisitBinaryOperator(BinaryOperator *E);
    Expr *VisitConditionalOperator(ConditionalOperator *E);
    Expr *VisitBinaryConditionalOperator(BinaryConditionalOperator *E);
    Expr *VisitOpaqueValueExpr(OpaqueValueExpr *E);
    Expr *VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E);
    Expr *VisitExpressionTraitExpr(ExpressionTraitExpr *E);
    Expr *VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    Expr *VisitCompoundAssignOperator(CompoundAssignOperator *E);
    Expr *VisitImplicitCastExpr(ImplicitCastExpr *E);
    Expr *VisitExplicitCastExpr(ExplicitCastExpr *E);
    Expr *VisitOffsetOfExpr(OffsetOfExpr *OE);
    Expr *VisitCXXThrowExpr(CXXThrowExpr *E);
    Expr *VisitCXXNoexceptExpr(CXXNoexceptExpr *E);
    Expr *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E);
    Expr *VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
    Expr *VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E);
    Expr *VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *CE);
    Expr *VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E);
    Expr *VisitCXXNewExpr(CXXNewExpr *CE);
    Expr *VisitCXXDeleteExpr(CXXDeleteExpr *E);
    Expr *VisitCXXConstructExpr(CXXConstructExpr *E);
    Expr *VisitCXXMemberCallExpr(CXXMemberCallExpr *E);
    Expr *VisitExprWithCleanups(ExprWithCleanups *EWC);
    Expr *VisitCXXThisExpr(CXXThisExpr *E);
    Expr *VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E);
    Expr *VisitMemberExpr(MemberExpr *E);
    Expr *VisitCallExpr(CallExpr *E);
    Expr *VisitInitListExpr(InitListExpr *E);
    Expr *VisitArrayInitLoopExpr(ArrayInitLoopExpr *E);
    Expr *VisitArrayInitIndexExpr(ArrayInitIndexExpr *E);
    Expr *VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E);
    Expr *VisitCXXNamedCastExpr(CXXNamedCastExpr *E);
    Expr *VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E);


    template<typename IIter, typename OIter>
    void ImportArray(IIter Ibegin, IIter Iend, OIter Obegin) {
      typedef typename std::remove_reference<decltype(*Obegin)>::type ItemT;
      ASTImporter &ImporterRef = Importer;
      std::transform(Ibegin, Iend, Obegin,
                     [&ImporterRef](ItemT From) -> ItemT {
                       return ImporterRef.Import(From);
                     });
    }

    template<typename IIter, typename OIter>
    bool ImportArrayChecked(IIter Ibegin, IIter Iend, OIter Obegin) {
      typedef typename std::remove_reference<decltype(**Obegin)>::type ItemT;
      ASTImporter &ImporterRef = Importer;
      bool Failed = false;
      std::transform(Ibegin, Iend, Obegin,
                     [&ImporterRef, &Failed](ItemT *From) -> ItemT * {
                       ItemT *To = cast_or_null<ItemT>(
                             ImporterRef.Import(From));
                       if (!To && From)
                         Failed = true;
                       return To;
                     });
      return Failed;
    }

    template<typename InContainerTy, typename OutContainerTy>
    bool ImportContainerChecked(const InContainerTy &InContainer,
                                OutContainerTy &OutContainer) {
      return ImportArrayChecked(InContainer.begin(), InContainer.end(),
                                OutContainer.begin());
    }

    template<typename InContainerTy, typename OIter>
    bool ImportArrayChecked(const InContainerTy &InContainer, OIter Obegin) {
      return ImportArrayChecked(InContainer.begin(), InContainer.end(), Obegin);
    }
  };
}

//----------------------------------------------------------------------------
// Import Types
//----------------------------------------------------------------------------

using namespace clang;

QualType ASTNodeImporter::VisitType(const Type *T) {
  Importer.FromDiag(SourceLocation(), diag::err_unsupported_ast_node)
    << T->getTypeClassName();
  return QualType();
}

QualType ASTNodeImporter::VisitAtomicType(const AtomicType *T){
  QualType UnderlyingType = Importer.Import(T->getValueType());
  if(UnderlyingType.isNull())
    return QualType();

  return Importer.getToContext().getAtomicType(UnderlyingType);
}

QualType ASTNodeImporter::VisitBuiltinType(const BuiltinType *T) {
  switch (T->getKind()) {
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix) \
  case BuiltinType::Id: \
    return Importer.getToContext().SingletonId;
#include "clang/Basic/OpenCLImageTypes.def"
#define SHARED_SINGLETON_TYPE(Expansion)
#define BUILTIN_TYPE(Id, SingletonId) \
  case BuiltinType::Id: return Importer.getToContext().SingletonId;
#include "clang/AST/BuiltinTypes.def"

  // FIXME: for Char16, Char32, and NullPtr, make sure that the "to"
  // context supports C++.

  // FIXME: for ObjCId, ObjCClass, and ObjCSel, make sure that the "to"
  // context supports ObjC.

  case BuiltinType::Char_U:
    // The context we're importing from has an unsigned 'char'. If we're 
    // importing into a context with a signed 'char', translate to 
    // 'unsigned char' instead.
    if (Importer.getToContext().getLangOpts().CharIsSigned)
      return Importer.getToContext().UnsignedCharTy;
    
    return Importer.getToContext().CharTy;

  case BuiltinType::Char_S:
    // The context we're importing from has an unsigned 'char'. If we're 
    // importing into a context with a signed 'char', translate to 
    // 'unsigned char' instead.
    if (!Importer.getToContext().getLangOpts().CharIsSigned)
      return Importer.getToContext().SignedCharTy;
    
    return Importer.getToContext().CharTy;

  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
    // FIXME: If not in C++, shall we translate to the C equivalent of
    // wchar_t?
    return Importer.getToContext().WCharTy;
  }

  llvm_unreachable("Invalid BuiltinType Kind!");
}

QualType ASTNodeImporter::VisitDecayedType(const DecayedType *T) {
  QualType OrigT = Importer.Import(T->getOriginalType());
  if (OrigT.isNull())
    return QualType();

  return Importer.getToContext().getDecayedType(OrigT);
}

QualType ASTNodeImporter::VisitComplexType(const ComplexType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getComplexType(ToElementType);
}

QualType ASTNodeImporter::VisitPointerType(const PointerType *T) {
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getPointerType(ToPointeeType);
}

QualType ASTNodeImporter::VisitBlockPointerType(const BlockPointerType *T) {
  // FIXME: Check for blocks support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getBlockPointerType(ToPointeeType);
}

QualType
ASTNodeImporter::VisitLValueReferenceType(const LValueReferenceType *T) {
  // FIXME: Check for C++ support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeTypeAsWritten());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getLValueReferenceType(ToPointeeType);
}

QualType
ASTNodeImporter::VisitRValueReferenceType(const RValueReferenceType *T) {
  // FIXME: Check for C++0x support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeTypeAsWritten());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getRValueReferenceType(ToPointeeType);  
}

QualType ASTNodeImporter::VisitMemberPointerType(const MemberPointerType *T) {
  // FIXME: Check for C++ support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  QualType ClassType = Importer.Import(QualType(T->getClass(), 0));
  return Importer.getToContext().getMemberPointerType(ToPointeeType, 
                                                      ClassType.getTypePtr());
}

QualType ASTNodeImporter::VisitConstantArrayType(const ConstantArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getConstantArrayType(ToElementType, 
                                                      T->getSize(),
                                                      T->getSizeModifier(),
                                               T->getIndexTypeCVRQualifiers());
}

QualType
ASTNodeImporter::VisitIncompleteArrayType(const IncompleteArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getIncompleteArrayType(ToElementType, 
                                                        T->getSizeModifier(),
                                                T->getIndexTypeCVRQualifiers());
}

QualType ASTNodeImporter::VisitVariableArrayType(const VariableArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();

  Expr *Size = Importer.Import(T->getSizeExpr());
  if (!Size)
    return QualType();
  
  SourceRange Brackets = Importer.Import(T->getBracketsRange());
  return Importer.getToContext().getVariableArrayType(ToElementType, Size,
                                                      T->getSizeModifier(),
                                                T->getIndexTypeCVRQualifiers(),
                                                      Brackets);
}

QualType ASTNodeImporter::VisitVectorType(const VectorType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getVectorType(ToElementType, 
                                               T->getNumElements(),
                                               T->getVectorKind());
}

QualType ASTNodeImporter::VisitExtVectorType(const ExtVectorType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getExtVectorType(ToElementType, 
                                                  T->getNumElements());
}

QualType
ASTNodeImporter::VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
  // FIXME: What happens if we're importing a function without a prototype 
  // into C++? Should we make it variadic?
  QualType ToResultType = Importer.Import(T->getReturnType());
  if (ToResultType.isNull())
    return QualType();

  return Importer.getToContext().getFunctionNoProtoType(ToResultType,
                                                        T->getExtInfo());
}

QualType ASTNodeImporter::VisitFunctionProtoType(const FunctionProtoType *T) {
  QualType ToResultType = Importer.Import(T->getReturnType());
  if (ToResultType.isNull())
    return QualType();
  
  // Import argument types
  SmallVector<QualType, 4> ArgTypes;
  for (const auto &A : T->param_types()) {
    QualType ArgType = Importer.Import(A);
    if (ArgType.isNull())
      return QualType();
    ArgTypes.push_back(ArgType);
  }
  
  // Import exception types
  SmallVector<QualType, 4> ExceptionTypes;
  for (const auto &E : T->exceptions()) {
    QualType ExceptionType = Importer.Import(E);
    if (ExceptionType.isNull())
      return QualType();
    ExceptionTypes.push_back(ExceptionType);
  }

  FunctionProtoType::ExtProtoInfo FromEPI = T->getExtProtoInfo();
  FunctionProtoType::ExtProtoInfo ToEPI;

  ToEPI.ExtInfo = FromEPI.ExtInfo;
  ToEPI.Variadic = FromEPI.Variadic;
  ToEPI.HasTrailingReturn = FromEPI.HasTrailingReturn;
  ToEPI.TypeQuals = FromEPI.TypeQuals;
  ToEPI.RefQualifier = FromEPI.RefQualifier;
  ToEPI.ExceptionSpec.Type = FromEPI.ExceptionSpec.Type;
  ToEPI.ExceptionSpec.Exceptions = ExceptionTypes;
  ToEPI.ExceptionSpec.NoexceptExpr =
      Importer.Import(FromEPI.ExceptionSpec.NoexceptExpr);
  ToEPI.ExceptionSpec.SourceDecl = cast_or_null<FunctionDecl>(
      Importer.Import(FromEPI.ExceptionSpec.SourceDecl));
  ToEPI.ExceptionSpec.SourceTemplate = cast_or_null<FunctionDecl>(
      Importer.Import(FromEPI.ExceptionSpec.SourceTemplate));

  return Importer.getToContext().getFunctionType(ToResultType, ArgTypes, ToEPI);
}

QualType ASTNodeImporter::VisitParenType(const ParenType *T) {
  QualType ToInnerType = Importer.Import(T->getInnerType());
  if (ToInnerType.isNull())
    return QualType();
    
  return Importer.getToContext().getParenType(ToInnerType);
}

QualType ASTNodeImporter::VisitTypedefType(const TypedefType *T) {
  TypedefNameDecl *ToDecl
             = dyn_cast_or_null<TypedefNameDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();
  
  return Importer.getToContext().getTypeDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitTypeOfExprType(const TypeOfExprType *T) {
  Expr *ToExpr = Importer.Import(T->getUnderlyingExpr());
  if (!ToExpr)
    return QualType();
  
  return Importer.getToContext().getTypeOfExprType(ToExpr);
}

QualType ASTNodeImporter::VisitTypeOfType(const TypeOfType *T) {
  QualType ToUnderlyingType = Importer.Import(T->getUnderlyingType());
  if (ToUnderlyingType.isNull())
    return QualType();
  
  return Importer.getToContext().getTypeOfType(ToUnderlyingType);
}

QualType ASTNodeImporter::VisitDecltypeType(const DecltypeType *T) {
  // FIXME: Make sure that the "to" context supports C++0x!
  Expr *ToExpr = Importer.Import(T->getUnderlyingExpr());
  if (!ToExpr)
    return QualType();
  
  QualType UnderlyingType = Importer.Import(T->getUnderlyingType());
  if (UnderlyingType.isNull())
    return QualType();

  return Importer.getToContext().getDecltypeType(ToExpr, UnderlyingType);
}

QualType ASTNodeImporter::VisitUnaryTransformType(const UnaryTransformType *T) {
  QualType ToBaseType = Importer.Import(T->getBaseType());
  QualType ToUnderlyingType = Importer.Import(T->getUnderlyingType());
  if (ToBaseType.isNull() || ToUnderlyingType.isNull())
    return QualType();

  return Importer.getToContext().getUnaryTransformType(ToBaseType,
                                                       ToUnderlyingType,
                                                       T->getUTTKind());
}

QualType ASTNodeImporter::VisitAutoType(const AutoType *T) {
  // FIXME: Make sure that the "to" context supports C++11!
  QualType FromDeduced = T->getDeducedType();
  QualType ToDeduced;
  if (!FromDeduced.isNull()) {
    ToDeduced = Importer.Import(FromDeduced);
    if (ToDeduced.isNull())
      return QualType();
  }
  
  return Importer.getToContext().getAutoType(ToDeduced, T->getKeyword(),
                                             /*IsDependent*/false);
}

QualType ASTNodeImporter::VisitInjectedClassNameType(
    const InjectedClassNameType *T) {
  CXXRecordDecl *D = cast_or_null<CXXRecordDecl>(Importer.Import(T->getDecl()));
  if (!D)
    return QualType();

  QualType InjType = Importer.Import(T->getInjectedSpecializationType());
  if (InjType.isNull())
    return QualType();

  // FIXME: ASTContext::getInjectedClassNameType is not suitable for AST reading
  // See comments in InjectedClassNameType definition for details
  // return Importer.getToContext().getInjectedClassNameType(D, InjType);
  enum {
    TypeAlignmentInBits = 4,
    TypeAlignment = 1 << TypeAlignmentInBits
  };

  return QualType(new (Importer.getToContext(), TypeAlignment)
                  InjectedClassNameType(D, InjType), 0);
}

QualType ASTNodeImporter::VisitRecordType(const RecordType *T) {
  RecordDecl *ToDecl
    = dyn_cast_or_null<RecordDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getTagDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitEnumType(const EnumType *T) {
  EnumDecl *ToDecl
    = dyn_cast_or_null<EnumDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getTagDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitAttributedType(const AttributedType *T) {
  QualType FromModifiedType = T->getModifiedType();
  QualType FromEquivalentType = T->getEquivalentType();
  QualType ToModifiedType;
  QualType ToEquivalentType;

  if (!FromModifiedType.isNull()) {
    ToModifiedType = Importer.Import(FromModifiedType);
    if (ToModifiedType.isNull())
      return QualType();
  }
  if (!FromEquivalentType.isNull()) {
    ToEquivalentType = Importer.Import(FromEquivalentType);
    if (ToEquivalentType.isNull())
      return QualType();
  }

  return Importer.getToContext().getAttributedType(T->getAttrKind(),
    ToModifiedType, ToEquivalentType);
}


QualType ASTNodeImporter::VisitTemplateTypeParmType(
    const TemplateTypeParmType *T) {
  TemplateTypeParmDecl *ParmDecl =
      cast_or_null<TemplateTypeParmDecl>(Importer.Import(T->getDecl()));
  if (!ParmDecl && T->getDecl())
    return QualType();

  return Importer.getToContext().getTemplateTypeParmType(
        T->getDepth(), T->getIndex(), T->isParameterPack(), ParmDecl);
}

QualType ASTNodeImporter::VisitSubstTemplateTypeParmType(
    const SubstTemplateTypeParmType *T) {
  const TemplateTypeParmType *Replaced =
      cast_or_null<TemplateTypeParmType>(Importer.Import(
        QualType(T->getReplacedParameter(), 0)).getTypePtr());
  if (!Replaced)
    return QualType();

  QualType Replacement = Importer.Import(T->getReplacementType());
  if (Replacement.isNull())
    return QualType();
  Replacement = Replacement.getCanonicalType();

  return Importer.getToContext().getSubstTemplateTypeParmType(
        Replaced, Replacement);
}

QualType ASTNodeImporter::VisitTemplateSpecializationType(
                                       const TemplateSpecializationType *T) {
  TemplateName ToTemplate = Importer.Import(T->getTemplateName());
  if (ToTemplate.isNull())
    return QualType();
  
  SmallVector<TemplateArgument, 2> ToTemplateArgs;
  if (ImportTemplateArguments(T->getArgs(), T->getNumArgs(), ToTemplateArgs))
    return QualType();
  
  QualType ToCanonType;
  if (!QualType(T, 0).isCanonical()) {
    QualType FromCanonType 
      = Importer.getFromContext().getCanonicalType(QualType(T, 0));
    ToCanonType =Importer.Import(FromCanonType);
    if (ToCanonType.isNull())
      return QualType();
  }
  return Importer.getToContext().getTemplateSpecializationType(ToTemplate, 
                                                               ToTemplateArgs,
                                                               ToCanonType);
}

QualType ASTNodeImporter::VisitElaboratedType(const ElaboratedType *T) {
  NestedNameSpecifier *ToQualifier = nullptr;
  // Note: the qualifier in an ElaboratedType is optional.
  if (T->getQualifier()) {
    ToQualifier = Importer.Import(T->getQualifier());
    if (!ToQualifier)
      return QualType();
  }

  QualType ToNamedType = Importer.Import(T->getNamedType());
  if (ToNamedType.isNull())
    return QualType();

  return Importer.getToContext().getElaboratedType(T->getKeyword(),
                                                   ToQualifier, ToNamedType);
}

QualType ASTNodeImporter::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  ObjCInterfaceDecl *Class
    = dyn_cast_or_null<ObjCInterfaceDecl>(Importer.Import(T->getDecl()));
  if (!Class)
    return QualType();

  return Importer.getToContext().getObjCInterfaceType(Class);
}

QualType ASTNodeImporter::VisitObjCObjectType(const ObjCObjectType *T) {
  QualType ToBaseType = Importer.Import(T->getBaseType());
  if (ToBaseType.isNull())
    return QualType();

  SmallVector<QualType, 4> TypeArgs;
  for (auto TypeArg : T->getTypeArgsAsWritten()) {
    QualType ImportedTypeArg = Importer.Import(TypeArg);
    if (ImportedTypeArg.isNull())
      return QualType();

    TypeArgs.push_back(ImportedTypeArg);
  }

  SmallVector<ObjCProtocolDecl *, 4> Protocols;
  for (auto *P : T->quals()) {
    ObjCProtocolDecl *Protocol
      = dyn_cast_or_null<ObjCProtocolDecl>(Importer.Import(P));
    if (!Protocol)
      return QualType();
    Protocols.push_back(Protocol);
  }

  return Importer.getToContext().getObjCObjectType(ToBaseType, TypeArgs,
                                                   Protocols,
                                                   T->isKindOfTypeAsWritten());
}

QualType
ASTNodeImporter::VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();

  return Importer.getToContext().getObjCObjectPointerType(ToPointeeType);
}

//----------------------------------------------------------------------------
// Import Declarations
//----------------------------------------------------------------------------
bool ASTNodeImporter::ImportDeclParts(NamedDecl *D, DeclContext *&DC, 
                                      DeclContext *&LexicalDC, 
                                      DeclarationName &Name, 
                                      NamedDecl *&ToD,
                                      SourceLocation &Loc) {
  // Import the context of this declaration.
  DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return true;
  
  LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return true;
  }
  
  // Import the name of this declaration.
  Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return true;
  
  // Import the location of this declaration.
  Loc = Importer.Import(D->getLocation());
  ToD = cast_or_null<NamedDecl>(Importer.GetAlreadyImportedOrNull(D));
  return false;
}

void ASTNodeImporter::ImportDefinitionIfNeeded(Decl *FromD, Decl *ToD) {
  if (!FromD)
    return;
  
  if (!ToD) {
    ToD = Importer.Import(FromD);
    if (!ToD)
      return;
  }
  
  if (RecordDecl *FromRecord = dyn_cast<RecordDecl>(FromD)) {
    if (RecordDecl *ToRecord = cast_or_null<RecordDecl>(ToD)) {
      if (FromRecord->getDefinition() && FromRecord->isCompleteDefinition() && !ToRecord->getDefinition()) {
        ImportDefinition(FromRecord, ToRecord);
      }
    }
    return;
  }

  if (EnumDecl *FromEnum = dyn_cast<EnumDecl>(FromD)) {
    if (EnumDecl *ToEnum = cast_or_null<EnumDecl>(ToD)) {
      if (FromEnum->getDefinition() && !ToEnum->getDefinition()) {
        ImportDefinition(FromEnum, ToEnum);
      }
    }
    return;
  }
}

void
ASTNodeImporter::ImportDeclarationNameLoc(const DeclarationNameInfo &From,
                                          DeclarationNameInfo& To) {
  // NOTE: To.Name and To.Loc are already imported.
  // We only have to import To.LocInfo.
  switch (To.getName().getNameKind()) {
  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXUsingDirective:
  case DeclarationName::CXXDeductionGuideName:
    return;

  case DeclarationName::CXXOperatorName: {
    SourceRange Range = From.getCXXOperatorNameRange();
    To.setCXXOperatorNameRange(Importer.Import(Range));
    return;
  }
  case DeclarationName::CXXLiteralOperatorName: {
    SourceLocation Loc = From.getCXXLiteralOperatorNameLoc();
    To.setCXXLiteralOperatorNameLoc(Importer.Import(Loc));
    return;
  }
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName: {
    TypeSourceInfo *FromTInfo = From.getNamedTypeInfo();
    To.setNamedTypeInfo(Importer.Import(FromTInfo));
    return;
  }
  }
  llvm_unreachable("Unknown name kind.");
}

void ASTNodeImporter::ImportDeclContext(DeclContext *FromDC, bool ForceImport) {  
  if (Importer.isMinimalImport() && !ForceImport) {
    Importer.ImportContext(FromDC);
    return;
  }
  
  for (auto *From : FromDC->decls())
    Importer.Import(From);
}

bool ASTNodeImporter::ImportDefinition(RecordDecl *From, RecordDecl *To, 
                                       ImportDefinitionKind Kind) {
  if (To->getDefinition() || To->isBeingDefined()) {
    if (Kind == IDK_Everything)
      ImportDeclContext(From, /*ForceImport=*/true);
    
    return false;
  }
  
  To->startDefinition();
  
  // Add base classes.
  if (CXXRecordDecl *ToCXX = dyn_cast<CXXRecordDecl>(To)) {
    CXXRecordDecl *FromCXX = cast<CXXRecordDecl>(From);

    struct CXXRecordDecl::DefinitionData &ToData = ToCXX->data();
    struct CXXRecordDecl::DefinitionData &FromData = FromCXX->data();
    ToData.UserDeclaredConstructor = FromData.UserDeclaredConstructor;
    ToData.UserDeclaredSpecialMembers = FromData.UserDeclaredSpecialMembers;
    ToData.Aggregate = FromData.Aggregate;
    ToData.PlainOldData = FromData.PlainOldData;
    ToData.Empty = FromData.Empty;
    ToData.Polymorphic = FromData.Polymorphic;
    ToData.Abstract = FromData.Abstract;
    ToData.IsStandardLayout = FromData.IsStandardLayout;
    ToData.HasNoNonEmptyBases = FromData.HasNoNonEmptyBases;
    ToData.HasPrivateFields = FromData.HasPrivateFields;
    ToData.HasProtectedFields = FromData.HasProtectedFields;
    ToData.HasPublicFields = FromData.HasPublicFields;
    ToData.HasMutableFields = FromData.HasMutableFields;
    ToData.HasVariantMembers = FromData.HasVariantMembers;
    ToData.HasOnlyCMembers = FromData.HasOnlyCMembers;
    ToData.HasInClassInitializer = FromData.HasInClassInitializer;
    ToData.HasUninitializedReferenceMember
      = FromData.HasUninitializedReferenceMember;
    ToData.HasUninitializedFields = FromData.HasUninitializedFields;
    ToData.HasInheritedConstructor = FromData.HasInheritedConstructor;
    ToData.HasInheritedAssignment = FromData.HasInheritedAssignment;
    ToData.NeedOverloadResolutionForMoveConstructor
      = FromData.NeedOverloadResolutionForMoveConstructor;
    ToData.NeedOverloadResolutionForMoveAssignment
      = FromData.NeedOverloadResolutionForMoveAssignment;
    ToData.NeedOverloadResolutionForDestructor
      = FromData.NeedOverloadResolutionForDestructor;
    ToData.DefaultedMoveConstructorIsDeleted
      = FromData.DefaultedMoveConstructorIsDeleted;
    ToData.DefaultedMoveAssignmentIsDeleted
      = FromData.DefaultedMoveAssignmentIsDeleted;
    ToData.DefaultedDestructorIsDeleted = FromData.DefaultedDestructorIsDeleted;
    ToData.HasTrivialSpecialMembers = FromData.HasTrivialSpecialMembers;
    ToData.HasIrrelevantDestructor = FromData.HasIrrelevantDestructor;
    ToData.HasConstexprNonCopyMoveConstructor
      = FromData.HasConstexprNonCopyMoveConstructor;
    ToData.HasDefaultedDefaultConstructor
      = FromData.HasDefaultedDefaultConstructor;
    ToData.DefaultedDefaultConstructorIsConstexpr
      = FromData.DefaultedDefaultConstructorIsConstexpr;
    ToData.HasConstexprDefaultConstructor
      = FromData.HasConstexprDefaultConstructor;
    ToData.HasNonLiteralTypeFieldsOrBases
      = FromData.HasNonLiteralTypeFieldsOrBases;
    // ComputedVisibleConversions not imported.
    ToData.UserProvidedDefaultConstructor
      = FromData.UserProvidedDefaultConstructor;
    ToData.DeclaredSpecialMembers = FromData.DeclaredSpecialMembers;
    ToData.ImplicitCopyConstructorCanHaveConstParamForVBase
      = FromData.ImplicitCopyConstructorCanHaveConstParamForVBase;
    ToData.ImplicitCopyConstructorCanHaveConstParamForNonVBase
      = FromData.ImplicitCopyConstructorCanHaveConstParamForNonVBase;
    ToData.ImplicitCopyAssignmentHasConstParam
      = FromData.ImplicitCopyAssignmentHasConstParam;
    ToData.HasDeclaredCopyConstructorWithConstParam
      = FromData.HasDeclaredCopyConstructorWithConstParam;
    ToData.HasDeclaredCopyAssignmentWithConstParam
      = FromData.HasDeclaredCopyAssignmentWithConstParam;
    ToData.IsLambda = FromData.IsLambda;

    SmallVector<CXXBaseSpecifier *, 4> Bases;
    for (const auto &Base1 : FromCXX->bases()) {
      QualType T = Importer.Import(Base1.getType());
      if (T.isNull())
        return true;

      SourceLocation EllipsisLoc;
      if (Base1.isPackExpansion())
        EllipsisLoc = Importer.Import(Base1.getEllipsisLoc());

      // Ensure that we have a definition for the base.
      ImportDefinitionIfNeeded(Base1.getType()->getAsCXXRecordDecl());
        
      Bases.push_back(
                    new (Importer.getToContext()) 
                      CXXBaseSpecifier(Importer.Import(Base1.getSourceRange()),
                                       Base1.isVirtual(),
                                       Base1.isBaseOfClass(),
                                       Base1.getAccessSpecifierAsWritten(),
                                   Importer.Import(Base1.getTypeSourceInfo()),
                                       EllipsisLoc));
    }
    if (!Bases.empty())
      ToCXX->setBases(Bases.data(), Bases.size());
  }
  
  if (shouldForceImportDeclContext(Kind))
    ImportDeclContext(From, /*ForceImport=*/true);
  
  To->completeDefinition();
  return false;
}

bool ASTNodeImporter::ImportDefinition(VarDecl *From, VarDecl *To,
                                       ImportDefinitionKind Kind) {
  if (To->getAnyInitializer())
    return false;

  // FIXME: Can we really import any initializer? Alternatively, we could force
  // ourselves to import every declaration of a variable and then only use
  // getInit() here.
  To->setInit(Importer.Import(const_cast<Expr *>(From->getAnyInitializer())));

  // FIXME: Other bits to merge?

  return false;
}

bool ASTNodeImporter::ImportDefinition(EnumDecl *From, EnumDecl *To, 
                                       ImportDefinitionKind Kind) {
  if (To->getDefinition() || To->isBeingDefined()) {
    if (Kind == IDK_Everything)
      ImportDeclContext(From, /*ForceImport=*/true);
    return false;
  }
  
  To->startDefinition();

  QualType T = Importer.Import(Importer.getFromContext().getTypeDeclType(From));
  if (T.isNull())
    return true;
  
  QualType ToPromotionType = Importer.Import(From->getPromotionType());
  if (ToPromotionType.isNull())
    return true;

  if (shouldForceImportDeclContext(Kind))
    ImportDeclContext(From, /*ForceImport=*/true);
  
  // FIXME: we might need to merge the number of positive or negative bits
  // if the enumerator lists don't match.
  To->completeDefinition(T, ToPromotionType,
                         From->getNumPositiveBits(),
                         From->getNumNegativeBits());
  return false;
}

TemplateParameterList *ASTNodeImporter::ImportTemplateParameterList(
                                                TemplateParameterList *Params) {
  SmallVector<NamedDecl *, 4> ToParams(Params->size());
  if (ImportContainerChecked(*Params, ToParams))
    return nullptr;
  
  Expr *ToRequiresClause;
  if (Expr *const R = Params->getRequiresClause()) {
    ToRequiresClause = Importer.Import(R);
    if (!ToRequiresClause)
      return nullptr;
  } else {
    ToRequiresClause = nullptr;
  }

  return TemplateParameterList::Create(Importer.getToContext(),
                                       Importer.Import(Params->getTemplateLoc()),
                                       Importer.Import(Params->getLAngleLoc()),
                                       ToParams,
                                       Importer.Import(Params->getRAngleLoc()),
                                       ToRequiresClause);
}

TemplateArgument 
ASTNodeImporter::ImportTemplateArgument(const TemplateArgument &From) {
  switch (From.getKind()) {
  case TemplateArgument::Null:
    return TemplateArgument();
     
  case TemplateArgument::Type: {
    QualType ToType = Importer.Import(From.getAsType());
    if (ToType.isNull())
      return TemplateArgument();
    return TemplateArgument(ToType);
  }
      
  case TemplateArgument::Integral: {
    QualType ToType = Importer.Import(From.getIntegralType());
    if (ToType.isNull())
      return TemplateArgument();
    return TemplateArgument(From, ToType);
  }

  case TemplateArgument::Declaration: {
    ValueDecl *To = cast_or_null<ValueDecl>(Importer.Import(From.getAsDecl()));
    QualType ToType = Importer.Import(From.getParamTypeForDecl());
    if (!To || ToType.isNull())
      return TemplateArgument();
    return TemplateArgument(To, ToType);
  }

  case TemplateArgument::NullPtr: {
    QualType ToType = Importer.Import(From.getNullPtrType());
    if (ToType.isNull())
      return TemplateArgument();
    return TemplateArgument(ToType, /*isNullPtr*/true);
  }

  case TemplateArgument::Template: {
    TemplateName ToTemplate = Importer.Import(From.getAsTemplate());
    if (ToTemplate.isNull())
      return TemplateArgument();
    
    return TemplateArgument(ToTemplate);
  }

  case TemplateArgument::TemplateExpansion: {
    TemplateName ToTemplate 
      = Importer.Import(From.getAsTemplateOrTemplatePattern());
    if (ToTemplate.isNull())
      return TemplateArgument();
    
    return TemplateArgument(ToTemplate, From.getNumTemplateExpansions());
  }

  case TemplateArgument::Expression:
    if (Expr *ToExpr = Importer.Import(From.getAsExpr()))
      return TemplateArgument(ToExpr);
    return TemplateArgument();
      
  case TemplateArgument::Pack: {
    SmallVector<TemplateArgument, 2> ToPack;
    ToPack.reserve(From.pack_size());
    if (ImportTemplateArguments(From.pack_begin(), From.pack_size(), ToPack))
      return TemplateArgument();

    return TemplateArgument(
        llvm::makeArrayRef(ToPack).copy(Importer.getToContext()));
  }
  }
  
  llvm_unreachable("Invalid template argument kind");
}

TemplateArgumentLoc ASTNodeImporter::ImportTemplateArgumentLoc(
    const TemplateArgumentLoc &TALoc, bool &Error) {
  Error = false;
  TemplateArgument Arg = ImportTemplateArgument(TALoc.getArgument());
  TemplateArgumentLocInfo FromInfo = TALoc.getLocInfo();
  TemplateArgumentLocInfo ToInfo;
  if (Arg.getKind() == TemplateArgument::Expression) {
    Expr *E = Importer.Import(FromInfo.getAsExpr());
    ToInfo = TemplateArgumentLocInfo(E);
    if (!E)
      Error = true;
  } else if (Arg.getKind() == TemplateArgument::Type) {
    if (TypeSourceInfo *TSI = Importer.Import(FromInfo.getAsTypeSourceInfo()))
      ToInfo = TemplateArgumentLocInfo(TSI);
    else
      Error = true;
  } else {
    ToInfo = TemplateArgumentLocInfo(
          Importer.Import(FromInfo.getTemplateQualifierLoc()),
          Importer.Import(FromInfo.getTemplateNameLoc()),
          Importer.Import(FromInfo.getTemplateEllipsisLoc()));
  }
  return TemplateArgumentLoc(Arg, ToInfo);
}

bool ASTNodeImporter::ImportTemplateArguments(const TemplateArgument *FromArgs,
                                              unsigned NumFromArgs,
                              SmallVectorImpl<TemplateArgument> &ToArgs) {
  for (unsigned I = 0; I != NumFromArgs; ++I) {
    TemplateArgument To = ImportTemplateArgument(FromArgs[I]);
    if (To.isNull() && !FromArgs[I].isNull())
      return true;
    
    ToArgs.push_back(To);
  }
  
  return false;
}

bool ASTNodeImporter::IsStructuralMatch(RecordDecl *FromRecord, 
                                        RecordDecl *ToRecord, bool Complain) {
  // Eliminate a potential failure point where we attempt to re-import
  // something we're trying to import while completing ToRecord.
  Decl *ToOrigin = Importer.GetOriginalDecl(ToRecord);
  if (ToOrigin) {
    RecordDecl *ToOriginRecord = dyn_cast<RecordDecl>(ToOrigin);
    if (ToOriginRecord)
      ToRecord = ToOriginRecord;
  }

  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   ToRecord->getASTContext(),
                                   Importer.getNonEquivalentDecls(),
                                   false, Complain);
  return Ctx.IsStructurallyEquivalent(FromRecord, ToRecord);
}

bool ASTNodeImporter::IsStructuralMatch(VarDecl *FromVar, VarDecl *ToVar,
                                        bool Complain) {
  StructuralEquivalenceContext Ctx(
      Importer.getFromContext(), Importer.getToContext(),
      Importer.getNonEquivalentDecls(), false, Complain);
  return Ctx.IsStructurallyEquivalent(FromVar, ToVar);
}

bool ASTNodeImporter::IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToEnum) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(FromEnum, ToEnum);
}

bool ASTNodeImporter::IsStructuralMatch(EnumConstantDecl *FromEC,
                                        EnumConstantDecl *ToEC)
{
  const llvm::APSInt &FromVal = FromEC->getInitVal();
  const llvm::APSInt &ToVal = ToEC->getInitVal();

  return FromVal.isSigned() == ToVal.isSigned() &&
         FromVal.getBitWidth() == ToVal.getBitWidth() &&
         FromVal == ToVal;
}

bool ASTNodeImporter::IsStructuralMatch(ClassTemplateDecl *From,
                                        ClassTemplateDecl *To) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(From, To);
}

bool ASTNodeImporter::IsStructuralMatch(VarTemplateDecl *From,
                                        VarTemplateDecl *To) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(From, To);
}

Decl *ASTNodeImporter::VisitDecl(Decl *D) {
  Importer.FromDiag(D->getLocation(), diag::err_unsupported_ast_node)
    << D->getDeclKindName();
  return nullptr;
}

Decl *ASTNodeImporter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  TranslationUnitDecl *ToD = 
    Importer.getToContext().getTranslationUnitDecl();
    
  Importer.Imported(D, ToD);
    
  return ToD;
}

Decl *ASTNodeImporter::VisitAccessSpecDecl(AccessSpecDecl *D) {

  SourceLocation Loc = Importer.Import(D->getLocation());
  SourceLocation ColonLoc = Importer.Import(D->getColonLoc());

  // Import the context of this declaration.
  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return nullptr;

  AccessSpecDecl *accessSpecDecl
    = AccessSpecDecl::Create(Importer.getToContext(), D->getAccess(),
                             DC, Loc, ColonLoc);

  if (!accessSpecDecl)
    return nullptr;

  // Lexical DeclContext and Semantic DeclContext
  // is always the same for the accessSpec.
  accessSpecDecl->setLexicalDeclContext(DC);
  DC->addDeclInternal(accessSpecDecl);

  return accessSpecDecl;
}

Decl *ASTNodeImporter::VisitStaticAssertDecl(StaticAssertDecl *D) {
  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return nullptr;

  DeclContext *LexicalDC = DC;

  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());

  Expr *AssertExpr = Importer.Import(D->getAssertExpr());
  if (!AssertExpr)
    return nullptr;

  StringLiteral *FromMsg = D->getMessage();
  StringLiteral *ToMsg = cast_or_null<StringLiteral>(Importer.Import(FromMsg));
  if (!ToMsg && FromMsg)
    return nullptr;

  StaticAssertDecl *ToD = StaticAssertDecl::Create(
        Importer.getToContext(), DC, Loc, AssertExpr, ToMsg,
        Importer.Import(D->getRParenLoc()), D->isFailed());

  ToD->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToD);
  Importer.Imported(D, ToD);
  return ToD;
}

Decl *ASTNodeImporter::VisitNamespaceDecl(NamespaceDecl *D) {
  // Import the major distinguishing characteristics of this namespace.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  NamespaceDecl *MergeWithNamespace = nullptr;
  if (!Name) {
    // This is an anonymous namespace. Adopt an existing anonymous
    // namespace if we can.
    // FIXME: Not testable.
    if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(DC))
      MergeWithNamespace = TU->getAnonymousNamespace();
    else
      MergeWithNamespace = cast<NamespaceDecl>(DC)->getAnonymousNamespace();
  } else {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(Decl::IDNS_Namespace))
        continue;
      
      if (NamespaceDecl *FoundNS = dyn_cast<NamespaceDecl>(FoundDecls[I])) {
        MergeWithNamespace = FoundNS;
        ConflictingDecls.clear();
        break;
      }
      
      ConflictingDecls.push_back(FoundDecls[I]);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, Decl::IDNS_Namespace,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
  }
  
  // Create the "to" namespace, if needed.
  NamespaceDecl *ToNamespace = MergeWithNamespace;
  if (!ToNamespace) {
    ToNamespace = NamespaceDecl::Create(Importer.getToContext(), DC,
                                        D->isInline(),
                                        Importer.Import(D->getLocStart()),
                                        Loc, Name.getAsIdentifierInfo(),
                                        /*PrevDecl=*/nullptr);
    ToNamespace->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToNamespace);
    
    // If this is an anonymous namespace, register it as the anonymous
    // namespace within its context.
    if (!Name) {
      if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(DC))
        TU->setAnonymousNamespace(ToNamespace);
      else
        cast<NamespaceDecl>(DC)->setAnonymousNamespace(ToNamespace);
    }
  }
  Importer.Imported(D, ToNamespace);
  
  ImportDeclContext(D);
  
  return ToNamespace;
}

Decl *ASTNodeImporter::VisitTypedefNameDecl(TypedefNameDecl *D, bool IsAlias) {
  // Import the major distinguishing characteristics of this typedef.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // If this typedef is not in block scope, determine whether we've
  // seen a typedef with the same name (that we can merge with) or any
  // other entity by that name (which name lookup could conflict with).
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;
      if (TypedefNameDecl *FoundTypedef =
            dyn_cast<TypedefNameDecl>(FoundDecls[I])) {
        if (Importer.IsStructurallyEquivalent(D->getUnderlyingType(),
                                            FoundTypedef->getUnderlyingType()))
          return Importer.Imported(D, FoundTypedef);
      }

      ConflictingDecls.push_back(FoundDecls[I]);
    }

    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return nullptr;
    }
  }

  // Import the underlying type of this typedef;
  QualType T = Importer.Import(D->getUnderlyingType());
  if (T.isNull())
    return nullptr;

  // Create the new typedef node.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  SourceLocation StartL = Importer.Import(D->getLocStart());
  TypedefNameDecl *ToTypedef;
  if (IsAlias)
    ToTypedef = TypeAliasDecl::Create(Importer.getToContext(), DC,
                                      StartL, Loc,
                                      Name.getAsIdentifierInfo(),
                                      TInfo);
  else
    ToTypedef = TypedefDecl::Create(Importer.getToContext(), DC,
                                    StartL, Loc,
                                    Name.getAsIdentifierInfo(),
                                    TInfo);

  ToTypedef->setAccess(D->getAccess());
  ToTypedef->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToTypedef);
  LexicalDC->addDeclInternal(ToTypedef);

  return ToTypedef;
}

Decl *ASTNodeImporter::VisitTypedefDecl(TypedefDecl *D) {
  return VisitTypedefNameDecl(D, /*IsAlias=*/false);
}

Decl *ASTNodeImporter::VisitTypeAliasDecl(TypeAliasDecl *D) {
  return VisitTypedefNameDecl(D, /*IsAlias=*/true);
}

Decl *ASTNodeImporter::VisitLabelDecl(LabelDecl *D) {
  // Import the major distinguishing characteristics of this label.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  assert(LexicalDC->isFunctionOrMethod());

  LabelDecl *ToLabel = D->isGnuLocal()
      ? LabelDecl::Create(Importer.getToContext(),
                          DC, Importer.Import(D->getLocation()),
                          Name.getAsIdentifierInfo(),
                          Importer.Import(D->getLocStart()))
      : LabelDecl::Create(Importer.getToContext(),
                          DC, Importer.Import(D->getLocation()),
                          Name.getAsIdentifierInfo());
  Importer.Imported(D, ToLabel);

  LabelStmt *Label = cast_or_null<LabelStmt>(Importer.Import(D->getStmt()));
  if (!Label)
    return nullptr;

  ToLabel->setStmt(Label);
  ToLabel->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToLabel);
  return ToLabel;
}

Decl *ASTNodeImporter::VisitEnumDecl(EnumDecl *D) {
  // Import the major distinguishing characteristics of this enum.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Figure out what enum name we're looking for.
  unsigned IDNS = Decl::IDNS_Tag;
  DeclarationName SearchName = Name;
  if (!SearchName && D->getTypedefNameForAnonDecl()) {
    SearchName = Importer.Import(D->getTypedefNameForAnonDecl()->getDeclName());
    IDNS = Decl::IDNS_Ordinary;
  } else if (Importer.getToContext().getLangOpts().CPlusPlus)
    IDNS |= Decl::IDNS_Ordinary;
  
  // We may already have an enum of the same name; try to find and match it.
  if (!DC->isFunctionOrMethod() && SearchName) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(SearchName, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;
      
      Decl *Found = FoundDecls[I];
      if (TypedefNameDecl *Typedef = dyn_cast<TypedefNameDecl>(Found)) {
        if (const TagType *Tag = Typedef->getUnderlyingType()->getAs<TagType>())
          Found = Tag->getDecl();
      }
      
      if (EnumDecl *FoundEnum = dyn_cast<EnumDecl>(Found)) {
        if (IsStructuralMatch(D, FoundEnum))
          return Importer.Imported(D, FoundEnum);
      }
      
      ConflictingDecls.push_back(FoundDecls[I]);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
  }
  
  // Create the enum declaration.
  EnumDecl *D2 = EnumDecl::Create(Importer.getToContext(), DC,
                                  Importer.Import(D->getLocStart()),
                                  Loc, Name.getAsIdentifierInfo(), nullptr,
                                  D->isScoped(), D->isScopedUsingClassTag(),
                                  D->isFixed());
  // Import the qualifier, if any.
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setAccess(D->getAccess());
  D2->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, D2);
  LexicalDC->addDeclInternal(D2);

  // Import the integer type.
  QualType ToIntegerType = Importer.Import(D->getIntegerType());
  if (ToIntegerType.isNull())
    return nullptr;
  D2->setIntegerType(ToIntegerType);
  
  // Import the definition
  if (D->isCompleteDefinition() && ImportDefinition(D, D2))
    return nullptr;

  return D2;
}

Decl *ASTNodeImporter::VisitRecordDecl(RecordDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  TagDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return nullptr;

    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Figure out what structure name we're looking for.
  unsigned IDNS = Decl::IDNS_Tag;
  DeclarationName SearchName = Name;
  if (!SearchName && D->getTypedefNameForAnonDecl()) {
    SearchName = Importer.Import(D->getTypedefNameForAnonDecl()->getDeclName());
    IDNS = Decl::IDNS_Ordinary;
  } else if (Importer.getToContext().getLangOpts().CPlusPlus)
    IDNS |= Decl::IDNS_Ordinary;

  // We may already have a record of the same name; try to find and match it.
  RecordDecl *AdoptDecl = nullptr;
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(SearchName, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;
      
      Decl *Found = FoundDecls[I];
      if (TypedefNameDecl *Typedef = dyn_cast<TypedefNameDecl>(Found)) {
        if (const TagType *Tag = Typedef->getUnderlyingType()->getAs<TagType>())
          Found = Tag->getDecl();
      }
      
      if (RecordDecl *FoundRecord = dyn_cast<RecordDecl>(Found)) {
        if (D->isAnonymousStructOrUnion() && 
            FoundRecord->isAnonymousStructOrUnion()) {
          // If both anonymous structs/unions are in a record context, make sure
          // they occur in the same location in the context records.
          if (Optional<unsigned> Index1 =
                  StructuralEquivalenceContext::findUntaggedStructOrUnionIndex(
                      D)) {
            if (Optional<unsigned> Index2 = StructuralEquivalenceContext::
                    findUntaggedStructOrUnionIndex(FoundRecord)) {
              if (*Index1 != *Index2)
                continue;
            }
          }
        }

        if (RecordDecl *FoundDef = FoundRecord->getDefinition()) {
          if ((SearchName && !D->isCompleteDefinition())
              || (D->isCompleteDefinition() &&
                  D->isAnonymousStructOrUnion()
                    == FoundDef->isAnonymousStructOrUnion() &&
                  IsStructuralMatch(D, FoundDef))) {
            // The record types structurally match, or the "from" translation
            // unit only had a forward declaration anyway; call it the same
            // function.
            // FIXME: For C++, we should also merge methods here.
            return Importer.Imported(D, FoundDef);
          }
        } else if (!D->isCompleteDefinition()) {
          // We have a forward declaration of this type, so adopt that forward
          // declaration rather than building a new one.
            
          // If one or both can be completed from external storage then try one
          // last time to complete and compare them before doing this.
            
          if (FoundRecord->hasExternalLexicalStorage() &&
              !FoundRecord->isCompleteDefinition())
            FoundRecord->getASTContext().getExternalSource()->CompleteType(FoundRecord);
          if (D->hasExternalLexicalStorage())
            D->getASTContext().getExternalSource()->CompleteType(D);
            
          if (FoundRecord->isCompleteDefinition() &&
              D->isCompleteDefinition() &&
              !IsStructuralMatch(D, FoundRecord))
            continue;
              
          AdoptDecl = FoundRecord;
          continue;
        } else if (!SearchName) {
          continue;
        }
      }
      
      ConflictingDecls.push_back(FoundDecls[I]);
    }
    
    if (!ConflictingDecls.empty() && SearchName) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
  }
  
  // Create the record declaration.
  RecordDecl *D2 = AdoptDecl;
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  if (!D2) {
    CXXRecordDecl *D2CXX = nullptr;
    if (CXXRecordDecl *DCXX = llvm::dyn_cast<CXXRecordDecl>(D)) {
      if (DCXX->isLambda()) {
        TypeSourceInfo *TInfo = Importer.Import(DCXX->getLambdaTypeInfo());
        D2CXX = CXXRecordDecl::CreateLambda(Importer.getToContext(),
                                            DC, TInfo, Loc,
                                            DCXX->isDependentLambda(),
                                            DCXX->isGenericLambda(),
                                            DCXX->getLambdaCaptureDefault());
        Decl *CDecl = Importer.Import(DCXX->getLambdaContextDecl());
        if (DCXX->getLambdaContextDecl() && !CDecl)
          return nullptr;
        D2CXX->setLambdaMangling(DCXX->getLambdaManglingNumber(), CDecl);
      } else if (DCXX->isInjectedClassName()) {                                                 
        // We have to be careful to do a similar dance to the one in                            
        // Sema::ActOnStartCXXMemberDeclarations                                                
        CXXRecordDecl *const PrevDecl = nullptr;                                                
        const bool DelayTypeCreation = true;                                                    
        D2CXX = CXXRecordDecl::Create(                                                          
            Importer.getToContext(), D->getTagKind(), DC, StartLoc, Loc,                        
            Name.getAsIdentifierInfo(), PrevDecl, DelayTypeCreation);                           
        Importer.getToContext().getTypeDeclType(                                                
            D2CXX, llvm::dyn_cast<CXXRecordDecl>(DC));                                          
      } else {
        D2CXX = CXXRecordDecl::Create(Importer.getToContext(),
                                      D->getTagKind(),
                                      DC, StartLoc, Loc,
                                      Name.getAsIdentifierInfo());
      }
      D2 = D2CXX;
      D2->setAccess(D->getAccess());
    } else {
      D2 = RecordDecl::Create(Importer.getToContext(), D->getTagKind(),
                              DC, StartLoc, Loc, Name.getAsIdentifierInfo());
    }
    
    D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
    D2->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(D2);
    if (D->isAnonymousStructOrUnion())
      D2->setAnonymousStructOrUnion(true);
  }
  
  Importer.Imported(D, D2);

  if (D->isCompleteDefinition() && ImportDefinition(D, D2, IDK_Default))
    return nullptr;

  return D2;
}

Decl *ASTNodeImporter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  // Import the major distinguishing characteristics of this enumerator.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  // Determine whether there are any other declarations with the same name and 
  // in the same context.
  if (!LexicalDC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;

      if (EnumConstantDecl *FoundEnumConstant
            = dyn_cast<EnumConstantDecl>(FoundDecls[I])) {
        if (IsStructuralMatch(D, FoundEnumConstant))
          return Importer.Imported(D, FoundEnumConstant);
      }

      ConflictingDecls.push_back(FoundDecls[I]);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return nullptr;
    }
  }
  
  Expr *Init = Importer.Import(D->getInitExpr());
  if (D->getInitExpr() && !Init)
    return nullptr;

  EnumConstantDecl *ToEnumerator
    = EnumConstantDecl::Create(Importer.getToContext(), cast<EnumDecl>(DC), Loc, 
                               Name.getAsIdentifierInfo(), T, 
                               Init, D->getInitVal());
  ToEnumerator->setAccess(D->getAccess());
  ToEnumerator->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToEnumerator);
  LexicalDC->addDeclInternal(ToEnumerator);
  return ToEnumerator;
}

Decl *ASTNodeImporter::VisitFunctionDecl(FunctionDecl *D) {
  // Import the major distinguishing characteristics of this function.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Try to find a function in our own ("to") context with the same name, same
  // type, and in the same context as the function we're importing.
  if (!LexicalDC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;

      if (FunctionDecl *FoundFunction = dyn_cast<FunctionDecl>(FoundDecls[I])) {
        if (FoundFunction->hasExternalFormalLinkage() &&
            D->hasExternalFormalLinkage()) {
          if (Importer.IsStructurallyEquivalent(D->getType(), 
                                                FoundFunction->getType())) {
            // FIXME: Actually try to merge the body and other attributes.
            return Importer.Imported(D, FoundFunction);
          }

          // FIXME: Check for overloading more carefully, e.g., by boosting
          // Sema::IsOverload out to the AST library.

          // Function overloading is okay in C++.
          if (Importer.getToContext().getLangOpts().CPlusPlus)
            continue;

          // Complain about inconsistent function types.
          Importer.ToDiag(Loc, diag::err_odr_function_type_inconsistent)
            << Name << D->getType() << FoundFunction->getType();
          Importer.ToDiag(FoundFunction->getLocation(), 
                          diag::note_odr_value_here)
            << FoundFunction->getType();
        }
      }

      ConflictingDecls.push_back(FoundDecls[I]);
    }

    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return nullptr;
    }    
  }

  DeclarationNameInfo NameInfo(Name, Loc);
  // Import additional name location/type info.
  ImportDeclarationNameLoc(D->getNameInfo(), NameInfo);

  QualType FromTy = D->getType();
  bool usedDifferentExceptionSpec = false;

  if (const FunctionProtoType *
        FromFPT = D->getType()->getAs<FunctionProtoType>()) {
    FunctionProtoType::ExtProtoInfo FromEPI = FromFPT->getExtProtoInfo();
    // FunctionProtoType::ExtProtoInfo's ExceptionSpecDecl can point to the
    // FunctionDecl that we are importing the FunctionProtoType for.
    // To avoid an infinite recursion when importing, create the FunctionDecl
    // with a simplified function type and update it afterwards.
    if (FromEPI.ExceptionSpec.SourceDecl ||
        FromEPI.ExceptionSpec.SourceTemplate ||
        FromEPI.ExceptionSpec.NoexceptExpr) {
      FunctionProtoType::ExtProtoInfo DefaultEPI;
      FromTy = Importer.getFromContext().getFunctionType(
          FromFPT->getReturnType(), FromFPT->getParamTypes(), DefaultEPI);
      usedDifferentExceptionSpec = true;
    }
  }

  // Import the type.
  QualType T = Importer.Import(FromTy);
  if (T.isNull())
    return nullptr;

  // Import the function parameters.
  SmallVector<ParmVarDecl *, 8> Parameters;
  for (auto P : D->parameters()) {
    ParmVarDecl *ToP = cast_or_null<ParmVarDecl>(Importer.Import(P));
    if (!ToP)
      return nullptr;

    Parameters.push_back(ToP);
  }
  
  // Create the imported function.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  FunctionDecl *ToFunction = nullptr;
  SourceLocation InnerLocStart = Importer.Import(D->getInnerLocStart());
  if (CXXConstructorDecl *FromConstructor = dyn_cast<CXXConstructorDecl>(D)) {
    ToFunction = CXXConstructorDecl::Create(Importer.getToContext(),
                                            cast<CXXRecordDecl>(DC),
                                            InnerLocStart,
                                            NameInfo, T, TInfo, 
                                            FromConstructor->isExplicit(),
                                            D->isInlineSpecified(), 
                                            D->isImplicit(),
                                            D->isConstexpr());
    if (unsigned NumInitializers = FromConstructor->getNumCtorInitializers()) {
      SmallVector<CXXCtorInitializer *, 4> CtorInitializers;
      for (CXXCtorInitializer *I : FromConstructor->inits()) {
        CXXCtorInitializer *ToI =
            cast_or_null<CXXCtorInitializer>(Importer.Import(I));
        if (!ToI && I)
          return nullptr;
        CtorInitializers.push_back(ToI);
      }
      CXXCtorInitializer **Memory =
          new (Importer.getToContext()) CXXCtorInitializer *[NumInitializers];
      std::copy(CtorInitializers.begin(), CtorInitializers.end(), Memory);
      CXXConstructorDecl *ToCtor = llvm::cast<CXXConstructorDecl>(ToFunction);
      ToCtor->setCtorInitializers(Memory);
      ToCtor->setNumCtorInitializers(NumInitializers);
    }
  } else if (isa<CXXDestructorDecl>(D)) {
    ToFunction = CXXDestructorDecl::Create(Importer.getToContext(),
                                           cast<CXXRecordDecl>(DC),
                                           InnerLocStart,
                                           NameInfo, T, TInfo,
                                           D->isInlineSpecified(),
                                           D->isImplicit());
  } else if (CXXConversionDecl *FromConversion
                                           = dyn_cast<CXXConversionDecl>(D)) {
    ToFunction = CXXConversionDecl::Create(Importer.getToContext(), 
                                           cast<CXXRecordDecl>(DC),
                                           InnerLocStart,
                                           NameInfo, T, TInfo,
                                           D->isInlineSpecified(),
                                           FromConversion->isExplicit(),
                                           D->isConstexpr(),
                                           Importer.Import(D->getLocEnd()));
  } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    ToFunction = CXXMethodDecl::Create(Importer.getToContext(), 
                                       cast<CXXRecordDecl>(DC),
                                       InnerLocStart,
                                       NameInfo, T, TInfo,
                                       Method->getStorageClass(),
                                       Method->isInlineSpecified(),
                                       D->isConstexpr(),
                                       Importer.Import(D->getLocEnd()));
  } else {
    ToFunction = FunctionDecl::Create(Importer.getToContext(), DC,
                                      InnerLocStart,
                                      NameInfo, T, TInfo, D->getStorageClass(),
                                      D->isInlineSpecified(),
                                      D->hasWrittenPrototype(),
                                      D->isConstexpr());
  }

  // Import the qualifier, if any.
  ToFunction->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  ToFunction->setAccess(D->getAccess());
  ToFunction->setLexicalDeclContext(LexicalDC);
  ToFunction->setVirtualAsWritten(D->isVirtualAsWritten());
  ToFunction->setTrivial(D->isTrivial());
  ToFunction->setPure(D->isPure());
  Importer.Imported(D, ToFunction);

  // Set the parameters.
  for (unsigned I = 0, N = Parameters.size(); I != N; ++I) {
    Parameters[I]->setOwningFunction(ToFunction);
    ToFunction->addDeclInternal(Parameters[I]);
  }
  ToFunction->setParams(Parameters);

  if (usedDifferentExceptionSpec) {
    // Update FunctionProtoType::ExtProtoInfo.
    QualType T = Importer.Import(D->getType());
    if (T.isNull())
      return nullptr;
    ToFunction->setType(T);
  }

  // Import the body, if any.
  if (Stmt *FromBody = D->getBody()) {
    if (Stmt *ToBody = Importer.Import(FromBody)) {
      ToFunction->setBody(ToBody);
    }
  }

  // FIXME: Other bits to merge?

  // Add this function to the lexical context.
  LexicalDC->addDeclInternal(ToFunction);

  return ToFunction;
}

Decl *ASTNodeImporter::VisitCXXMethodDecl(CXXMethodDecl *D) {
  return VisitFunctionDecl(D);
}

Decl *ASTNodeImporter::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *ASTNodeImporter::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *ASTNodeImporter::VisitCXXConversionDecl(CXXConversionDecl *D) {
  return VisitCXXMethodDecl(D);
}

static unsigned getFieldIndex(Decl *F) {
  RecordDecl *Owner = dyn_cast<RecordDecl>(F->getDeclContext());
  if (!Owner)
    return 0;

  unsigned Index = 1;
  for (const auto *D : Owner->noload_decls()) {
    if (D == F)
      return Index;

    if (isa<FieldDecl>(*D) || isa<IndirectFieldDecl>(*D))
      ++Index;
  }

  return Index;
}

Decl *ASTNodeImporter::VisitFieldDecl(FieldDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Determine whether we've already imported this field. 
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (FieldDecl *FoundField = dyn_cast<FieldDecl>(FoundDecls[I])) {
      // For anonymous fields, match up by index.
      if (!Name && getFieldIndex(D) != getFieldIndex(FoundField))
        continue;

      if (Importer.IsStructurallyEquivalent(D->getType(),
                                            FoundField->getType())) {
        Importer.Imported(D, FoundField);
        return FoundField;
      }

      Importer.ToDiag(Loc, diag::err_odr_field_type_inconsistent)
        << Name << D->getType() << FoundField->getType();
      Importer.ToDiag(FoundField->getLocation(), diag::note_odr_value_here)
        << FoundField->getType();
      return nullptr;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  Expr *BitWidth = Importer.Import(D->getBitWidth());
  if (!BitWidth && D->getBitWidth())
    return nullptr;

  FieldDecl *ToField = FieldDecl::Create(Importer.getToContext(), DC,
                                         Importer.Import(D->getInnerLocStart()),
                                         Loc, Name.getAsIdentifierInfo(),
                                         T, TInfo, BitWidth, D->isMutable(),
                                         D->getInClassInitStyle());
  ToField->setAccess(D->getAccess());
  ToField->setLexicalDeclContext(LexicalDC);
  if (Expr *FromInitializer = D->getInClassInitializer()) {
    Expr *ToInitializer = Importer.Import(FromInitializer);
    if (ToInitializer)
      ToField->setInClassInitializer(ToInitializer);
    else
      return nullptr;
  }
  ToField->setImplicit(D->isImplicit());
  Importer.Imported(D, ToField);
  LexicalDC->addDeclInternal(ToField);
  return ToField;
}

Decl *ASTNodeImporter::VisitIndirectFieldDecl(IndirectFieldDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Determine whether we've already imported this field. 
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (IndirectFieldDecl *FoundField 
                                = dyn_cast<IndirectFieldDecl>(FoundDecls[I])) {
      // For anonymous indirect fields, match up by index.
      if (!Name && getFieldIndex(D) != getFieldIndex(FoundField))
        continue;

      if (Importer.IsStructurallyEquivalent(D->getType(),
                                            FoundField->getType(),
                                            !Name.isEmpty())) {
        Importer.Imported(D, FoundField);
        return FoundField;
      }

      // If there are more anonymous fields to check, continue.
      if (!Name && I < N-1)
        continue;

      Importer.ToDiag(Loc, diag::err_odr_field_type_inconsistent)
        << Name << D->getType() << FoundField->getType();
      Importer.ToDiag(FoundField->getLocation(), diag::note_odr_value_here)
        << FoundField->getType();
      return nullptr;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  NamedDecl **NamedChain =
    new (Importer.getToContext())NamedDecl*[D->getChainingSize()];

  unsigned i = 0;
  for (auto *PI : D->chain()) {
    Decl *D = Importer.Import(PI);
    if (!D)
      return nullptr;
    NamedChain[i++] = cast<NamedDecl>(D);
  }

  IndirectFieldDecl *ToIndirectField = IndirectFieldDecl::Create(
      Importer.getToContext(), DC, Loc, Name.getAsIdentifierInfo(), T,
      {NamedChain, D->getChainingSize()});

  for (const auto *Attr : D->attrs())
    ToIndirectField->addAttr(Attr->clone(Importer.getToContext()));

  ToIndirectField->setAccess(D->getAccess());
  ToIndirectField->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToIndirectField);
  LexicalDC->addDeclInternal(ToIndirectField);
  return ToIndirectField;
}

Decl *ASTNodeImporter::VisitFriendDecl(FriendDecl *D) {
  // Import the major distinguishing characteristics of a declaration.
  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  DeclContext *LexicalDC = D->getDeclContext() == D->getLexicalDeclContext()
      ? DC : Importer.ImportContext(D->getLexicalDeclContext());
  if (!DC || !LexicalDC)
    return nullptr;

  // Determine whether we've already imported this decl.
  // FriendDecl is not a NamedDecl so we cannot use localUncachedLookup.
  auto *RD = cast<CXXRecordDecl>(DC);
  FriendDecl *ImportedFriend = RD->getFirstFriend();
  StructuralEquivalenceContext Context(
      Importer.getFromContext(), Importer.getToContext(),
      Importer.getNonEquivalentDecls(), false, false);

  while (ImportedFriend) {
    if (D->getFriendDecl() && ImportedFriend->getFriendDecl()) {
      if (Context.IsStructurallyEquivalent(D->getFriendDecl(),
                                           ImportedFriend->getFriendDecl()))
        return Importer.Imported(D, ImportedFriend);

    } else if (D->getFriendType() && ImportedFriend->getFriendType()) {
      if (Importer.IsStructurallyEquivalent(
            D->getFriendType()->getType(),
            ImportedFriend->getFriendType()->getType(), true))
        return Importer.Imported(D, ImportedFriend);
    }
    ImportedFriend = ImportedFriend->getNextFriend();
  }

  // Not found. Create it.
  FriendDecl::FriendUnion ToFU;
  if (NamedDecl *FriendD = D->getFriendDecl())
    ToFU = cast_or_null<NamedDecl>(Importer.Import(FriendD));
  else
    ToFU = Importer.Import(D->getFriendType());
  if (!ToFU)
    return nullptr;

  SmallVector<TemplateParameterList *, 1> ToTPLists(D->NumTPLists);
  TemplateParameterList **FromTPLists =
      D->getTrailingObjects<TemplateParameterList *>();
  for (unsigned I = 0; I < D->NumTPLists; I++) {
    TemplateParameterList *List = ImportTemplateParameterList(FromTPLists[I]);
    if (!List)
      return nullptr;
    ToTPLists[I] = List;
  }

  FriendDecl *FrD = FriendDecl::Create(Importer.getToContext(), DC,
                                       Importer.Import(D->getLocation()),
                                       ToFU, Importer.Import(D->getFriendLoc()),
                                       ToTPLists);

  Importer.Imported(D, FrD);
  RD->pushFriendDecl(FrD);

  FrD->setAccess(D->getAccess());
  FrD->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(FrD);
  return FrD;
}

Decl *ASTNodeImporter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  // Import the major distinguishing characteristics of an ivar.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Determine whether we've already imported this ivar
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (ObjCIvarDecl *FoundIvar = dyn_cast<ObjCIvarDecl>(FoundDecls[I])) {
      if (Importer.IsStructurallyEquivalent(D->getType(),
                                            FoundIvar->getType())) {
        Importer.Imported(D, FoundIvar);
        return FoundIvar;
      }

      Importer.ToDiag(Loc, diag::err_odr_ivar_type_inconsistent)
        << Name << D->getType() << FoundIvar->getType();
      Importer.ToDiag(FoundIvar->getLocation(), diag::note_odr_value_here)
        << FoundIvar->getType();
      return nullptr;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  Expr *BitWidth = Importer.Import(D->getBitWidth());
  if (!BitWidth && D->getBitWidth())
    return nullptr;

  ObjCIvarDecl *ToIvar = ObjCIvarDecl::Create(Importer.getToContext(),
                                              cast<ObjCContainerDecl>(DC),
                                       Importer.Import(D->getInnerLocStart()),
                                              Loc, Name.getAsIdentifierInfo(),
                                              T, TInfo, D->getAccessControl(),
                                              BitWidth, D->getSynthesize());
  ToIvar->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToIvar);
  LexicalDC->addDeclInternal(ToIvar);
  return ToIvar;
  
}

Decl *ASTNodeImporter::VisitVarDecl(VarDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Try to find a variable in our own ("to") context with the same name and
  // in the same context as the variable we're importing.
  if (D->isFileVarDecl()) {
    VarDecl *MergeWithVar = nullptr;
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;

      if (VarDecl *FoundVar = dyn_cast<VarDecl>(FoundDecls[I])) {
        // We have found a variable that we may need to merge with. Check it.
        if (FoundVar->hasExternalFormalLinkage() &&
            D->hasExternalFormalLinkage()) {
          if (Importer.IsStructurallyEquivalent(D->getType(),
                                                FoundVar->getType())) {
            MergeWithVar = FoundVar;
            break;
          }

          const ArrayType *FoundArray
            = Importer.getToContext().getAsArrayType(FoundVar->getType());
          const ArrayType *TArray
            = Importer.getToContext().getAsArrayType(D->getType());
          if (FoundArray && TArray) {
            if (isa<IncompleteArrayType>(FoundArray) &&
                isa<ConstantArrayType>(TArray)) {
              // Import the type.
              QualType T = Importer.Import(D->getType());
              if (T.isNull())
                return nullptr;

              FoundVar->setType(T);
              MergeWithVar = FoundVar;
              break;
            } else if (isa<IncompleteArrayType>(TArray) &&
                       isa<ConstantArrayType>(FoundArray)) {
              MergeWithVar = FoundVar;
              break;
            }
          }

          Importer.ToDiag(Loc, diag::err_odr_variable_type_inconsistent)
            << Name << D->getType() << FoundVar->getType();
          Importer.ToDiag(FoundVar->getLocation(), diag::note_odr_value_here)
            << FoundVar->getType();
        }
      }
      
      ConflictingDecls.push_back(FoundDecls[I]);
    }

    if (MergeWithVar) {
      // An equivalent variable with external linkage has been found. Link 
      // the two declarations, then merge them.
      Importer.Imported(D, MergeWithVar);
      
      if (VarDecl *DDef = D->getDefinition()) {
        if (VarDecl *ExistingDef = MergeWithVar->getDefinition()) {
          Importer.ToDiag(ExistingDef->getLocation(), 
                          diag::err_odr_variable_multiple_def)
            << Name;
          Importer.FromDiag(DDef->getLocation(), diag::note_odr_defined_here);
        } else {
          Expr *Init = Importer.Import(DDef->getInit());
          MergeWithVar->setInit(Init);
          if (DDef->isInitKnownICE()) {
            EvaluatedStmt *Eval = MergeWithVar->ensureEvaluatedStmt();
            Eval->CheckedICE = true;
            Eval->IsICE = DDef->isInitICE();
          }
        }
      }
      
      return MergeWithVar;
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return nullptr;
    }
  }
    
  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  // Create the imported variable.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  VarDecl *ToVar = VarDecl::Create(Importer.getToContext(), DC,
                                   Importer.Import(D->getInnerLocStart()),
                                   Loc, Name.getAsIdentifierInfo(),
                                   T, TInfo,
                                   D->getStorageClass());
  ToVar->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  ToVar->setAccess(D->getAccess());
  ToVar->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToVar);
  LexicalDC->addDeclInternal(ToVar);

  if (!D->isFileVarDecl() &&
      D->isUsed())
    ToVar->setIsUsed();

  // Merge the initializer.
  if (ImportDefinition(D, ToVar))
    return nullptr;

  if (D->isConstexpr())
    ToVar->setConstexpr(true);

  return ToVar;
}

Decl *ASTNodeImporter::VisitImplicitParamDecl(ImplicitParamDecl *D) {
  // Parameters are created in the translation unit's context, then moved
  // into the function declaration's context afterward.
  DeclContext *DC = Importer.getToContext().getTranslationUnitDecl();
  
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return nullptr;

  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import the parameter's type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  // Create the imported parameter.
  ImplicitParamDecl *ToParm
    = ImplicitParamDecl::Create(Importer.getToContext(), DC,
                                Loc, Name.getAsIdentifierInfo(),
                                T);
  return Importer.Imported(D, ToParm);
}

Decl *ASTNodeImporter::VisitParmVarDecl(ParmVarDecl *D) {
  // Parameters are created in the translation unit's context, then moved
  // into the function declaration's context afterward.
  DeclContext *DC = Importer.getToContext().getTranslationUnitDecl();
  
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return nullptr;

  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import the parameter's type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  // Create the imported parameter.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  ParmVarDecl *ToParm = ParmVarDecl::Create(Importer.getToContext(), DC,
                                     Importer.Import(D->getInnerLocStart()),
                                            Loc, Name.getAsIdentifierInfo(),
                                            T, TInfo, D->getStorageClass(),
                                            /*DefaultArg*/ nullptr);

  // Set the default argument.
  ToParm->setHasInheritedDefaultArg(D->hasInheritedDefaultArg());
  ToParm->setKNRPromoted(D->isKNRPromoted());

  Expr *ToDefArg = nullptr;
  Expr *FromDefArg = nullptr;
  if (D->hasUninstantiatedDefaultArg()) {
    FromDefArg = D->getUninstantiatedDefaultArg();
    ToDefArg = Importer.Import(FromDefArg);
    ToParm->setUninstantiatedDefaultArg(ToDefArg);
  } else if (D->hasUnparsedDefaultArg()) {
    ToParm->setUnparsedDefaultArg();
  } else if (D->hasDefaultArg()) {
    FromDefArg = D->getDefaultArg();
    ToDefArg = Importer.Import(FromDefArg);
    ToParm->setDefaultArg(ToDefArg);
  }
  if (FromDefArg && !ToDefArg)
    return nullptr;

  if (D->isUsed())
    ToParm->setIsUsed();

  return Importer.Imported(D, ToParm);
}

Decl *ASTNodeImporter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  // Import the major distinguishing characteristics of a method.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (ObjCMethodDecl *FoundMethod = dyn_cast<ObjCMethodDecl>(FoundDecls[I])) {
      if (FoundMethod->isInstanceMethod() != D->isInstanceMethod())
        continue;

      // Check return types.
      if (!Importer.IsStructurallyEquivalent(D->getReturnType(),
                                             FoundMethod->getReturnType())) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_result_type_inconsistent)
            << D->isInstanceMethod() << Name << D->getReturnType()
            << FoundMethod->getReturnType();
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return nullptr;
      }

      // Check the number of parameters.
      if (D->param_size() != FoundMethod->param_size()) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_num_params_inconsistent)
          << D->isInstanceMethod() << Name
          << D->param_size() << FoundMethod->param_size();
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return nullptr;
      }

      // Check parameter types.
      for (ObjCMethodDecl::param_iterator P = D->param_begin(),
             PEnd = D->param_end(), FoundP = FoundMethod->param_begin();
           P != PEnd; ++P, ++FoundP) {
        if (!Importer.IsStructurallyEquivalent((*P)->getType(),
                                               (*FoundP)->getType())) {
          Importer.FromDiag((*P)->getLocation(),
                            diag::err_odr_objc_method_param_type_inconsistent)
            << D->isInstanceMethod() << Name
            << (*P)->getType() << (*FoundP)->getType();
          Importer.ToDiag((*FoundP)->getLocation(), diag::note_odr_value_here)
            << (*FoundP)->getType();
          return nullptr;
        }
      }

      // Check variadic/non-variadic.
      // Check the number of parameters.
      if (D->isVariadic() != FoundMethod->isVariadic()) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_variadic_inconsistent)
          << D->isInstanceMethod() << Name;
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return nullptr;
      }

      // FIXME: Any other bits we need to merge?
      return Importer.Imported(D, FoundMethod);
    }
  }

  // Import the result type.
  QualType ResultTy = Importer.Import(D->getReturnType());
  if (ResultTy.isNull())
    return nullptr;

  TypeSourceInfo *ReturnTInfo = Importer.Import(D->getReturnTypeSourceInfo());

  ObjCMethodDecl *ToMethod = ObjCMethodDecl::Create(
      Importer.getToContext(), Loc, Importer.Import(D->getLocEnd()),
      Name.getObjCSelector(), ResultTy, ReturnTInfo, DC, D->isInstanceMethod(),
      D->isVariadic(), D->isPropertyAccessor(), D->isImplicit(), D->isDefined(),
      D->getImplementationControl(), D->hasRelatedResultType());

  // FIXME: When we decide to merge method definitions, we'll need to
  // deal with implicit parameters.

  // Import the parameters
  SmallVector<ParmVarDecl *, 5> ToParams;
  for (auto *FromP : D->parameters()) {
    ParmVarDecl *ToP = cast_or_null<ParmVarDecl>(Importer.Import(FromP));
    if (!ToP)
      return nullptr;

    ToParams.push_back(ToP);
  }
  
  // Set the parameters.
  for (unsigned I = 0, N = ToParams.size(); I != N; ++I) {
    ToParams[I]->setOwningFunction(ToMethod);
    ToMethod->addDeclInternal(ToParams[I]);
  }
  SmallVector<SourceLocation, 12> SelLocs;
  D->getSelectorLocs(SelLocs);
  ToMethod->setMethodParams(Importer.getToContext(), ToParams, SelLocs); 

  ToMethod->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToMethod);
  LexicalDC->addDeclInternal(ToMethod);
  return ToMethod;
}

Decl *ASTNodeImporter::VisitObjCTypeParamDecl(ObjCTypeParamDecl *D) {
  // Import the major distinguishing characteristics of a category.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  TypeSourceInfo *BoundInfo = Importer.Import(D->getTypeSourceInfo());
  if (!BoundInfo)
    return nullptr;

  ObjCTypeParamDecl *Result = ObjCTypeParamDecl::Create(
                                Importer.getToContext(), DC,
                                D->getVariance(),
                                Importer.Import(D->getVarianceLoc()),
                                D->getIndex(),
                                Importer.Import(D->getLocation()),
                                Name.getAsIdentifierInfo(),
                                Importer.Import(D->getColonLoc()),
                                BoundInfo);
  Importer.Imported(D, Result);
  Result->setLexicalDeclContext(LexicalDC);
  return Result;
}

Decl *ASTNodeImporter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  // Import the major distinguishing characteristics of a category.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  ObjCInterfaceDecl *ToInterface
    = cast_or_null<ObjCInterfaceDecl>(Importer.Import(D->getClassInterface()));
  if (!ToInterface)
    return nullptr;

  // Determine if we've already encountered this category.
  ObjCCategoryDecl *MergeWithCategory
    = ToInterface->FindCategoryDeclaration(Name.getAsIdentifierInfo());
  ObjCCategoryDecl *ToCategory = MergeWithCategory;
  if (!ToCategory) {
    ToCategory = ObjCCategoryDecl::Create(Importer.getToContext(), DC,
                                          Importer.Import(D->getAtStartLoc()),
                                          Loc, 
                                       Importer.Import(D->getCategoryNameLoc()), 
                                          Name.getAsIdentifierInfo(),
                                          ToInterface,
                                          /*TypeParamList=*/nullptr,
                                       Importer.Import(D->getIvarLBraceLoc()),
                                       Importer.Import(D->getIvarRBraceLoc()));
    ToCategory->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToCategory);
    Importer.Imported(D, ToCategory);
    // Import the type parameter list after calling Imported, to avoid
    // loops when bringing in their DeclContext.
    ToCategory->setTypeParamList(ImportObjCTypeParamList(
                                   D->getTypeParamList()));
    
    // Import protocols
    SmallVector<ObjCProtocolDecl *, 4> Protocols;
    SmallVector<SourceLocation, 4> ProtocolLocs;
    ObjCCategoryDecl::protocol_loc_iterator FromProtoLoc
      = D->protocol_loc_begin();
    for (ObjCCategoryDecl::protocol_iterator FromProto = D->protocol_begin(),
                                          FromProtoEnd = D->protocol_end();
         FromProto != FromProtoEnd;
         ++FromProto, ++FromProtoLoc) {
      ObjCProtocolDecl *ToProto
        = cast_or_null<ObjCProtocolDecl>(Importer.Import(*FromProto));
      if (!ToProto)
        return nullptr;
      Protocols.push_back(ToProto);
      ProtocolLocs.push_back(Importer.Import(*FromProtoLoc));
    }
    
    // FIXME: If we're merging, make sure that the protocol list is the same.
    ToCategory->setProtocolList(Protocols.data(), Protocols.size(),
                                ProtocolLocs.data(), Importer.getToContext());
    
  } else {
    Importer.Imported(D, ToCategory);
  }
  
  // Import all of the members of this category.
  ImportDeclContext(D);
 
  // If we have an implementation, import it as well.
  if (D->getImplementation()) {
    ObjCCategoryImplDecl *Impl
      = cast_or_null<ObjCCategoryImplDecl>(
                                       Importer.Import(D->getImplementation()));
    if (!Impl)
      return nullptr;

    ToCategory->setImplementation(Impl);
  }
  
  return ToCategory;
}

bool ASTNodeImporter::ImportDefinition(ObjCProtocolDecl *From, 
                                       ObjCProtocolDecl *To,
                                       ImportDefinitionKind Kind) {
  if (To->getDefinition()) {
    if (shouldForceImportDeclContext(Kind))
      ImportDeclContext(From);
    return false;
  }

  // Start the protocol definition
  To->startDefinition();
  
  // Import protocols
  SmallVector<ObjCProtocolDecl *, 4> Protocols;
  SmallVector<SourceLocation, 4> ProtocolLocs;
  ObjCProtocolDecl::protocol_loc_iterator 
  FromProtoLoc = From->protocol_loc_begin();
  for (ObjCProtocolDecl::protocol_iterator FromProto = From->protocol_begin(),
                                        FromProtoEnd = From->protocol_end();
       FromProto != FromProtoEnd;
       ++FromProto, ++FromProtoLoc) {
    ObjCProtocolDecl *ToProto
      = cast_or_null<ObjCProtocolDecl>(Importer.Import(*FromProto));
    if (!ToProto)
      return true;
    Protocols.push_back(ToProto);
    ProtocolLocs.push_back(Importer.Import(*FromProtoLoc));
  }
  
  // FIXME: If we're merging, make sure that the protocol list is the same.
  To->setProtocolList(Protocols.data(), Protocols.size(),
                      ProtocolLocs.data(), Importer.getToContext());

  if (shouldForceImportDeclContext(Kind)) {
    // Import all of the members of this protocol.
    ImportDeclContext(From, /*ForceImport=*/true);
  }
  return false;
}

Decl *ASTNodeImporter::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  // If this protocol has a definition in the translation unit we're coming 
  // from, but this particular declaration is not that definition, import the
  // definition and map to that.
  ObjCProtocolDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return nullptr;

    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of a protocol.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  ObjCProtocolDecl *MergeWithProtocol = nullptr;
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (!FoundDecls[I]->isInIdentifierNamespace(Decl::IDNS_ObjCProtocol))
      continue;
    
    if ((MergeWithProtocol = dyn_cast<ObjCProtocolDecl>(FoundDecls[I])))
      break;
  }
  
  ObjCProtocolDecl *ToProto = MergeWithProtocol;
  if (!ToProto) {
    ToProto = ObjCProtocolDecl::Create(Importer.getToContext(), DC,
                                       Name.getAsIdentifierInfo(), Loc,
                                       Importer.Import(D->getAtStartLoc()),
                                       /*PrevDecl=*/nullptr);
    ToProto->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToProto);
  }
    
  Importer.Imported(D, ToProto);

  if (D->isThisDeclarationADefinition() && ImportDefinition(D, ToProto))
    return nullptr;

  return ToProto;
}

Decl *ASTNodeImporter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  DeclContext *LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());

  SourceLocation ExternLoc = Importer.Import(D->getExternLoc());
  SourceLocation LangLoc = Importer.Import(D->getLocation());

  bool HasBraces = D->hasBraces();
 
  LinkageSpecDecl *ToLinkageSpec =
    LinkageSpecDecl::Create(Importer.getToContext(),
                            DC,
                            ExternLoc,
                            LangLoc,
                            D->getLanguage(),
                            HasBraces);

  if (HasBraces) {
    SourceLocation RBraceLoc = Importer.Import(D->getRBraceLoc());
    ToLinkageSpec->setRBraceLoc(RBraceLoc);
  }

  ToLinkageSpec->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToLinkageSpec);

  Importer.Imported(D, ToLinkageSpec);

  return ToLinkageSpec;
}

bool ASTNodeImporter::ImportDefinition(ObjCInterfaceDecl *From, 
                                       ObjCInterfaceDecl *To,
                                       ImportDefinitionKind Kind) {
  if (To->getDefinition()) {
    // Check consistency of superclass.
    ObjCInterfaceDecl *FromSuper = From->getSuperClass();
    if (FromSuper) {
      FromSuper = cast_or_null<ObjCInterfaceDecl>(Importer.Import(FromSuper));
      if (!FromSuper)
        return true;
    }
    
    ObjCInterfaceDecl *ToSuper = To->getSuperClass();    
    if ((bool)FromSuper != (bool)ToSuper ||
        (FromSuper && !declaresSameEntity(FromSuper, ToSuper))) {
      Importer.ToDiag(To->getLocation(), 
                      diag::err_odr_objc_superclass_inconsistent)
        << To->getDeclName();
      if (ToSuper)
        Importer.ToDiag(To->getSuperClassLoc(), diag::note_odr_objc_superclass)
          << To->getSuperClass()->getDeclName();
      else
        Importer.ToDiag(To->getLocation(), 
                        diag::note_odr_objc_missing_superclass);
      if (From->getSuperClass())
        Importer.FromDiag(From->getSuperClassLoc(), 
                          diag::note_odr_objc_superclass)
        << From->getSuperClass()->getDeclName();
      else
        Importer.FromDiag(From->getLocation(), 
                          diag::note_odr_objc_missing_superclass);        
    }
    
    if (shouldForceImportDeclContext(Kind))
      ImportDeclContext(From);
    return false;
  }
  
  // Start the definition.
  To->startDefinition();
  
  // If this class has a superclass, import it.
  if (From->getSuperClass()) {
    TypeSourceInfo *SuperTInfo = Importer.Import(From->getSuperClassTInfo());
    if (!SuperTInfo)
      return true;

    To->setSuperClass(SuperTInfo);
  }
  
  // Import protocols
  SmallVector<ObjCProtocolDecl *, 4> Protocols;
  SmallVector<SourceLocation, 4> ProtocolLocs;
  ObjCInterfaceDecl::protocol_loc_iterator 
  FromProtoLoc = From->protocol_loc_begin();
  
  for (ObjCInterfaceDecl::protocol_iterator FromProto = From->protocol_begin(),
                                         FromProtoEnd = From->protocol_end();
       FromProto != FromProtoEnd;
       ++FromProto, ++FromProtoLoc) {
    ObjCProtocolDecl *ToProto
      = cast_or_null<ObjCProtocolDecl>(Importer.Import(*FromProto));
    if (!ToProto)
      return true;
    Protocols.push_back(ToProto);
    ProtocolLocs.push_back(Importer.Import(*FromProtoLoc));
  }
  
  // FIXME: If we're merging, make sure that the protocol list is the same.
  To->setProtocolList(Protocols.data(), Protocols.size(),
                      ProtocolLocs.data(), Importer.getToContext());
  
  // Import categories. When the categories themselves are imported, they'll
  // hook themselves into this interface.
  for (auto *Cat : From->known_categories())
    Importer.Import(Cat);
  
  // If we have an @implementation, import it as well.
  if (From->getImplementation()) {
    ObjCImplementationDecl *Impl = cast_or_null<ObjCImplementationDecl>(
                                     Importer.Import(From->getImplementation()));
    if (!Impl)
      return true;
    
    To->setImplementation(Impl);
  }

  if (shouldForceImportDeclContext(Kind)) {
    // Import all of the members of this class.
    ImportDeclContext(From, /*ForceImport=*/true);
  }
  return false;
}

ObjCTypeParamList *
ASTNodeImporter::ImportObjCTypeParamList(ObjCTypeParamList *list) {
  if (!list)
    return nullptr;

  SmallVector<ObjCTypeParamDecl *, 4> toTypeParams;
  for (auto fromTypeParam : *list) {
    auto toTypeParam = cast_or_null<ObjCTypeParamDecl>(
                         Importer.Import(fromTypeParam));
    if (!toTypeParam)
      return nullptr;

    toTypeParams.push_back(toTypeParam);
  }

  return ObjCTypeParamList::create(Importer.getToContext(),
                                   Importer.Import(list->getLAngleLoc()),
                                   toTypeParams,
                                   Importer.Import(list->getRAngleLoc()));
}

Decl *ASTNodeImporter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // If this class has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  ObjCInterfaceDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return nullptr;

    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of an @interface.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Look for an existing interface with the same name.
  ObjCInterfaceDecl *MergeWithIface = nullptr;
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (!FoundDecls[I]->isInIdentifierNamespace(Decl::IDNS_Ordinary))
      continue;
    
    if ((MergeWithIface = dyn_cast<ObjCInterfaceDecl>(FoundDecls[I])))
      break;
  }
  
  // Create an interface declaration, if one does not already exist.
  ObjCInterfaceDecl *ToIface = MergeWithIface;
  if (!ToIface) {
    ToIface = ObjCInterfaceDecl::Create(Importer.getToContext(), DC,
                                        Importer.Import(D->getAtStartLoc()),
                                        Name.getAsIdentifierInfo(),
                                        /*TypeParamList=*/nullptr,
                                        /*PrevDecl=*/nullptr, Loc,
                                        D->isImplicitInterfaceDecl());
    ToIface->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToIface);
  }
  Importer.Imported(D, ToIface);
  // Import the type parameter list after calling Imported, to avoid
  // loops when bringing in their DeclContext.
  ToIface->setTypeParamList(ImportObjCTypeParamList(
                              D->getTypeParamListAsWritten()));
  
  if (D->isThisDeclarationADefinition() && ImportDefinition(D, ToIface))
    return nullptr;

  return ToIface;
}

Decl *ASTNodeImporter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  ObjCCategoryDecl *Category = cast_or_null<ObjCCategoryDecl>(
                                        Importer.Import(D->getCategoryDecl()));
  if (!Category)
    return nullptr;

  ObjCCategoryImplDecl *ToImpl = Category->getImplementation();
  if (!ToImpl) {
    DeclContext *DC = Importer.ImportContext(D->getDeclContext());
    if (!DC)
      return nullptr;

    SourceLocation CategoryNameLoc = Importer.Import(D->getCategoryNameLoc());
    ToImpl = ObjCCategoryImplDecl::Create(Importer.getToContext(), DC,
                                          Importer.Import(D->getIdentifier()),
                                          Category->getClassInterface(),
                                          Importer.Import(D->getLocation()),
                                          Importer.Import(D->getAtStartLoc()),
                                          CategoryNameLoc);
    
    DeclContext *LexicalDC = DC;
    if (D->getDeclContext() != D->getLexicalDeclContext()) {
      LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
      if (!LexicalDC)
        return nullptr;

      ToImpl->setLexicalDeclContext(LexicalDC);
    }
    
    LexicalDC->addDeclInternal(ToImpl);
    Category->setImplementation(ToImpl);
  }
  
  Importer.Imported(D, ToImpl);
  ImportDeclContext(D);
  return ToImpl;
}

Decl *ASTNodeImporter::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  // Find the corresponding interface.
  ObjCInterfaceDecl *Iface = cast_or_null<ObjCInterfaceDecl>(
                                       Importer.Import(D->getClassInterface()));
  if (!Iface)
    return nullptr;

  // Import the superclass, if any.
  ObjCInterfaceDecl *Super = nullptr;
  if (D->getSuperClass()) {
    Super = cast_or_null<ObjCInterfaceDecl>(
                                          Importer.Import(D->getSuperClass()));
    if (!Super)
      return nullptr;
  }

  ObjCImplementationDecl *Impl = Iface->getImplementation();
  if (!Impl) {
    // We haven't imported an implementation yet. Create a new @implementation
    // now.
    Impl = ObjCImplementationDecl::Create(Importer.getToContext(),
                                  Importer.ImportContext(D->getDeclContext()),
                                          Iface, Super,
                                          Importer.Import(D->getLocation()),
                                          Importer.Import(D->getAtStartLoc()),
                                          Importer.Import(D->getSuperClassLoc()),
                                          Importer.Import(D->getIvarLBraceLoc()),
                                          Importer.Import(D->getIvarRBraceLoc()));
    
    if (D->getDeclContext() != D->getLexicalDeclContext()) {
      DeclContext *LexicalDC
        = Importer.ImportContext(D->getLexicalDeclContext());
      if (!LexicalDC)
        return nullptr;
      Impl->setLexicalDeclContext(LexicalDC);
    }
    
    // Associate the implementation with the class it implements.
    Iface->setImplementation(Impl);
    Importer.Imported(D, Iface->getImplementation());
  } else {
    Importer.Imported(D, Iface->getImplementation());

    // Verify that the existing @implementation has the same superclass.
    if ((Super && !Impl->getSuperClass()) ||
        (!Super && Impl->getSuperClass()) ||
        (Super && Impl->getSuperClass() &&
         !declaresSameEntity(Super->getCanonicalDecl(),
                             Impl->getSuperClass()))) {
      Importer.ToDiag(Impl->getLocation(),
                      diag::err_odr_objc_superclass_inconsistent)
        << Iface->getDeclName();
      // FIXME: It would be nice to have the location of the superclass
      // below.
      if (Impl->getSuperClass())
        Importer.ToDiag(Impl->getLocation(),
                        diag::note_odr_objc_superclass)
        << Impl->getSuperClass()->getDeclName();
      else
        Importer.ToDiag(Impl->getLocation(),
                        diag::note_odr_objc_missing_superclass);
      if (D->getSuperClass())
        Importer.FromDiag(D->getLocation(),
                          diag::note_odr_objc_superclass)
        << D->getSuperClass()->getDeclName();
      else
        Importer.FromDiag(D->getLocation(),
                          diag::note_odr_objc_missing_superclass);
      return nullptr;
    }
  }
    
  // Import all of the members of this @implementation.
  ImportDeclContext(D);

  return Impl;
}

Decl *ASTNodeImporter::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  // Import the major distinguishing characteristics of an @property.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // Check whether we have already imported this property.
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (ObjCPropertyDecl *FoundProp
                                = dyn_cast<ObjCPropertyDecl>(FoundDecls[I])) {
      // Check property types.
      if (!Importer.IsStructurallyEquivalent(D->getType(),
                                             FoundProp->getType())) {
        Importer.ToDiag(Loc, diag::err_odr_objc_property_type_inconsistent)
          << Name << D->getType() << FoundProp->getType();
        Importer.ToDiag(FoundProp->getLocation(), diag::note_odr_value_here)
          << FoundProp->getType();
        return nullptr;
      }

      // FIXME: Check property attributes, getters, setters, etc.?

      // Consider these properties to be equivalent.
      Importer.Imported(D, FoundProp);
      return FoundProp;
    }
  }

  // Import the type.
  TypeSourceInfo *TSI = Importer.Import(D->getTypeSourceInfo());
  if (!TSI)
    return nullptr;

  // Create the new property.
  ObjCPropertyDecl *ToProperty
    = ObjCPropertyDecl::Create(Importer.getToContext(), DC, Loc,
                               Name.getAsIdentifierInfo(), 
                               Importer.Import(D->getAtLoc()),
                               Importer.Import(D->getLParenLoc()),
                               Importer.Import(D->getType()),
                               TSI,
                               D->getPropertyImplementation());
  Importer.Imported(D, ToProperty);
  ToProperty->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToProperty);

  ToProperty->setPropertyAttributes(D->getPropertyAttributes());
  ToProperty->setPropertyAttributesAsWritten(
                                      D->getPropertyAttributesAsWritten());
  ToProperty->setGetterName(Importer.Import(D->getGetterName()),
                            Importer.Import(D->getGetterNameLoc()));
  ToProperty->setSetterName(Importer.Import(D->getSetterName()),
                            Importer.Import(D->getSetterNameLoc()));
  ToProperty->setGetterMethodDecl(
     cast_or_null<ObjCMethodDecl>(Importer.Import(D->getGetterMethodDecl())));
  ToProperty->setSetterMethodDecl(
     cast_or_null<ObjCMethodDecl>(Importer.Import(D->getSetterMethodDecl())));
  ToProperty->setPropertyIvarDecl(
       cast_or_null<ObjCIvarDecl>(Importer.Import(D->getPropertyIvarDecl())));
  return ToProperty;
}

Decl *ASTNodeImporter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  ObjCPropertyDecl *Property = cast_or_null<ObjCPropertyDecl>(
                                        Importer.Import(D->getPropertyDecl()));
  if (!Property)
    return nullptr;

  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return nullptr;

  // Import the lexical declaration context.
  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return nullptr;
  }

  ObjCImplDecl *InImpl = dyn_cast<ObjCImplDecl>(LexicalDC);
  if (!InImpl)
    return nullptr;

  // Import the ivar (for an @synthesize).
  ObjCIvarDecl *Ivar = nullptr;
  if (D->getPropertyIvarDecl()) {
    Ivar = cast_or_null<ObjCIvarDecl>(
                                    Importer.Import(D->getPropertyIvarDecl()));
    if (!Ivar)
      return nullptr;
  }

  ObjCPropertyImplDecl *ToImpl
    = InImpl->FindPropertyImplDecl(Property->getIdentifier(),
                                   Property->getQueryKind());
  if (!ToImpl) {    
    ToImpl = ObjCPropertyImplDecl::Create(Importer.getToContext(), DC,
                                          Importer.Import(D->getLocStart()),
                                          Importer.Import(D->getLocation()),
                                          Property,
                                          D->getPropertyImplementation(),
                                          Ivar, 
                                  Importer.Import(D->getPropertyIvarDeclLoc()));
    ToImpl->setLexicalDeclContext(LexicalDC);
    Importer.Imported(D, ToImpl);
    LexicalDC->addDeclInternal(ToImpl);
  } else {
    // Check that we have the same kind of property implementation (@synthesize
    // vs. @dynamic).
    if (D->getPropertyImplementation() != ToImpl->getPropertyImplementation()) {
      Importer.ToDiag(ToImpl->getLocation(), 
                      diag::err_odr_objc_property_impl_kind_inconsistent)
        << Property->getDeclName() 
        << (ToImpl->getPropertyImplementation() 
                                              == ObjCPropertyImplDecl::Dynamic);
      Importer.FromDiag(D->getLocation(),
                        diag::note_odr_objc_property_impl_kind)
        << D->getPropertyDecl()->getDeclName()
        << (D->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic);
      return nullptr;
    }
    
    // For @synthesize, check that we have the same 
    if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize &&
        Ivar != ToImpl->getPropertyIvarDecl()) {
      Importer.ToDiag(ToImpl->getPropertyIvarDeclLoc(), 
                      diag::err_odr_objc_synthesize_ivar_inconsistent)
        << Property->getDeclName()
        << ToImpl->getPropertyIvarDecl()->getDeclName()
        << Ivar->getDeclName();
      Importer.FromDiag(D->getPropertyIvarDeclLoc(), 
                        diag::note_odr_objc_synthesize_ivar_here)
        << D->getPropertyIvarDecl()->getDeclName();
      return nullptr;
    }
    
    // Merge the existing implementation with the new implementation.
    Importer.Imported(D, ToImpl);
  }
  
  return ToImpl;
}

Decl *ASTNodeImporter::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  // For template arguments, we adopt the translation unit as our declaration
  // context. This context will be fixed when the actual template declaration
  // is created.
  
  // FIXME: Import default argument.
  return TemplateTypeParmDecl::Create(Importer.getToContext(),
                              Importer.getToContext().getTranslationUnitDecl(),
                                      Importer.Import(D->getLocStart()),
                                      Importer.Import(D->getLocation()),
                                      D->getDepth(),
                                      D->getIndex(), 
                                      Importer.Import(D->getIdentifier()),
                                      D->wasDeclaredWithTypename(),
                                      D->isParameterPack());
}

Decl *
ASTNodeImporter::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return nullptr;

  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());

  // Import the type of this declaration.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return nullptr;

  // Import type-source information.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  if (D->getTypeSourceInfo() && !TInfo)
    return nullptr;

  // FIXME: Import default argument.
  
  return NonTypeTemplateParmDecl::Create(Importer.getToContext(),
                               Importer.getToContext().getTranslationUnitDecl(),
                                         Importer.Import(D->getInnerLocStart()),
                                         Loc, D->getDepth(), D->getPosition(),
                                         Name.getAsIdentifierInfo(),
                                         T, D->isParameterPack(), TInfo);
}

Decl *
ASTNodeImporter::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return nullptr;

  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import template parameters.
  TemplateParameterList *TemplateParams
    = ImportTemplateParameterList(D->getTemplateParameters());
  if (!TemplateParams)
    return nullptr;

  // FIXME: Import default argument.
  
  return TemplateTemplateParmDecl::Create(Importer.getToContext(), 
                              Importer.getToContext().getTranslationUnitDecl(), 
                                          Loc, D->getDepth(), D->getPosition(),
                                          D->isParameterPack(),
                                          Name.getAsIdentifierInfo(), 
                                          TemplateParams);
}

Decl *ASTNodeImporter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  CXXRecordDecl *Definition 
    = cast_or_null<CXXRecordDecl>(D->getTemplatedDecl()->getDefinition());
  if (Definition && Definition != D->getTemplatedDecl()) {
    Decl *ImportedDef
      = Importer.Import(Definition->getDescribedClassTemplate());
    if (!ImportedDef)
      return nullptr;

    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this class template.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // We may already have a template of the same name; try to find and match it.
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    SmallVector<NamedDecl *, 2> FoundDecls;
    DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(Decl::IDNS_Ordinary))
        continue;
      
      Decl *Found = FoundDecls[I];
      if (ClassTemplateDecl *FoundTemplate 
                                        = dyn_cast<ClassTemplateDecl>(Found)) {
        if (IsStructuralMatch(D, FoundTemplate)) {
          // The class templates structurally match; call it the same template.
          // FIXME: We may be filling in a forward declaration here. Handle
          // this case!
          Importer.Imported(D->getTemplatedDecl(), 
                            FoundTemplate->getTemplatedDecl());
          return Importer.Imported(D, FoundTemplate);
        }         
      }
      
      ConflictingDecls.push_back(FoundDecls[I]);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, Decl::IDNS_Ordinary,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
    
    if (!Name)
      return nullptr;
  }

  CXXRecordDecl *DTemplated = D->getTemplatedDecl();
  
  // Create the declaration that is being templated.
  // Create the declaration that is being templated.
  CXXRecordDecl *D2Templated = cast_or_null<CXXRecordDecl>(
        Importer.Import(DTemplated));
  if (!D2Templated)
    return nullptr;

  // Resolve possible cyclic import.
  if (Decl *AlreadyImported = Importer.GetAlreadyImportedOrNull(D))
    return AlreadyImported;

  // Create the class template declaration itself.
  TemplateParameterList *TemplateParams
    = ImportTemplateParameterList(D->getTemplateParameters());
  if (!TemplateParams)
    return nullptr;

  ClassTemplateDecl *D2 = ClassTemplateDecl::Create(Importer.getToContext(), DC, 
                                                    Loc, Name, TemplateParams, 
                                                    D2Templated);
  D2Templated->setDescribedClassTemplate(D2);    
  
  D2->setAccess(D->getAccess());
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);
  
  // Note the relationship between the class templates.
  Importer.Imported(D, D2);
  Importer.Imported(DTemplated, D2Templated);

  if (DTemplated->isCompleteDefinition() &&
      !D2Templated->isCompleteDefinition()) {
    // FIXME: Import definition!
  }
  
  return D2;
}

Decl *ASTNodeImporter::VisitClassTemplateSpecializationDecl(
                                          ClassTemplateSpecializationDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  TagDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return nullptr;

    return Importer.Imported(D, ImportedDef);
  }

  ClassTemplateDecl *ClassTemplate
    = cast_or_null<ClassTemplateDecl>(Importer.Import(
                                                 D->getSpecializedTemplate()));
  if (!ClassTemplate)
    return nullptr;

  // Import the context of this declaration.
  DeclContext *DC = ClassTemplate->getDeclContext();
  if (!DC)
    return nullptr;

  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return nullptr;
  }
  
  // Import the location of this declaration.
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  SourceLocation IdLoc = Importer.Import(D->getLocation());

  // Import template arguments.
  SmallVector<TemplateArgument, 2> TemplateArgs;
  if (ImportTemplateArguments(D->getTemplateArgs().data(), 
                              D->getTemplateArgs().size(),
                              TemplateArgs))
    return nullptr;

  // Try to find an existing specialization with these template arguments.
  void *InsertPos = nullptr;
  ClassTemplateSpecializationDecl *D2
    = ClassTemplate->findSpecialization(TemplateArgs, InsertPos);
  if (D2) {
    // We already have a class template specialization with these template
    // arguments.
    
    // FIXME: Check for specialization vs. instantiation errors.
    
    if (RecordDecl *FoundDef = D2->getDefinition()) {
      if (!D->isCompleteDefinition() || IsStructuralMatch(D, FoundDef)) {
        // The record types structurally match, or the "from" translation
        // unit only had a forward declaration anyway; call it the same
        // function.
        return Importer.Imported(D, FoundDef);
      }
    }
  } else {
    // Create a new specialization.
    if (ClassTemplatePartialSpecializationDecl *PartialSpec =
        dyn_cast<ClassTemplatePartialSpecializationDecl>(D)) {

      // Import TemplateArgumentListInfo
      TemplateArgumentListInfo ToTAInfo;
      auto &ASTTemplateArgs = *PartialSpec->getTemplateArgsAsWritten();
      for (unsigned I = 0, E = ASTTemplateArgs.NumTemplateArgs; I < E; ++I) {
        bool Error = false;
        auto ToLoc = ImportTemplateArgumentLoc(ASTTemplateArgs[I], Error);
        if (Error)
          return nullptr;
        ToTAInfo.addArgument(ToLoc);
      }

      QualType CanonInjType = Importer.Import(
            PartialSpec->getInjectedSpecializationType());
      if (CanonInjType.isNull())
        return nullptr;
      CanonInjType = CanonInjType.getCanonicalType();

      TemplateParameterList *ToTPList = ImportTemplateParameterList(
            PartialSpec->getTemplateParameters());
      if (!ToTPList && PartialSpec->getTemplateParameters())
        return nullptr;

      D2 = ClassTemplatePartialSpecializationDecl::Create(
            Importer.getToContext(), D->getTagKind(), DC, StartLoc, IdLoc,
            ToTPList, ClassTemplate,
            llvm::makeArrayRef(TemplateArgs.data(), TemplateArgs.size()),
            ToTAInfo, CanonInjType, nullptr);

    } else {
      D2 = ClassTemplateSpecializationDecl::Create(Importer.getToContext(),
                                                   D->getTagKind(), DC,
                                                   StartLoc, IdLoc,
                                                   ClassTemplate,
                                                   TemplateArgs,
                                                   /*PrevDecl=*/nullptr);
    }

    D2->setSpecializationKind(D->getSpecializationKind());

    // Add this specialization to the class template.
    ClassTemplate->AddSpecialization(D2, InsertPos);
    
    // Import the qualifier, if any.
    D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));

    Importer.Imported(D, D2);

    if (auto *TSI = D->getTypeAsWritten()) {
      TypeSourceInfo *TInfo = Importer.Import(TSI);
      if (!TInfo)
        return nullptr;
      D2->setTypeAsWritten(TInfo);
      D2->setTemplateKeywordLoc(Importer.Import(D->getTemplateKeywordLoc()));
      D2->setExternLoc(Importer.Import(D->getExternLoc()));
    }

    SourceLocation POI = Importer.Import(D->getPointOfInstantiation());
    if (POI.isValid())
      D2->setPointOfInstantiation(POI);
    else if (D->getPointOfInstantiation().isValid())
      return nullptr;

    D2->setTemplateSpecializationKind(D->getTemplateSpecializationKind());

    // Add the specialization to this context.
    D2->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(D2);
  }
  Importer.Imported(D, D2);
  if (D->isCompleteDefinition() && ImportDefinition(D, D2))
    return nullptr;

  return D2;
}

Decl *ASTNodeImporter::VisitVarTemplateDecl(VarTemplateDecl *D) {
  // If this variable has a definition in the translation unit we're coming
  // from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  VarDecl *Definition =
      cast_or_null<VarDecl>(D->getTemplatedDecl()->getDefinition());
  if (Definition && Definition != D->getTemplatedDecl()) {
    Decl *ImportedDef = Importer.Import(Definition->getDescribedVarTemplate());
    if (!ImportedDef)
      return nullptr;

    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of this variable template.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return nullptr;
  if (ToD)
    return ToD;

  // We may already have a template of the same name; try to find and match it.
  assert(!DC->isFunctionOrMethod() &&
         "Variable templates cannot be declared at function scope");
  SmallVector<NamedDecl *, 4> ConflictingDecls;
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->getRedeclContext()->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (!FoundDecls[I]->isInIdentifierNamespace(Decl::IDNS_Ordinary))
      continue;

    Decl *Found = FoundDecls[I];
    if (VarTemplateDecl *FoundTemplate = dyn_cast<VarTemplateDecl>(Found)) {
      if (IsStructuralMatch(D, FoundTemplate)) {
        // The variable templates structurally match; call it the same template.
        Importer.Imported(D->getTemplatedDecl(),
                          FoundTemplate->getTemplatedDecl());
        return Importer.Imported(D, FoundTemplate);
      }
    }

    ConflictingDecls.push_back(FoundDecls[I]);
  }

  if (!ConflictingDecls.empty()) {
    Name = Importer.HandleNameConflict(Name, DC, Decl::IDNS_Ordinary,
                                       ConflictingDecls.data(),
                                       ConflictingDecls.size());
  }

  if (!Name)
    return nullptr;

  VarDecl *DTemplated = D->getTemplatedDecl();

  // Import the type.
  QualType T = Importer.Import(DTemplated->getType());
  if (T.isNull())
    return nullptr;

  // Create the declaration that is being templated.
  SourceLocation StartLoc = Importer.Import(DTemplated->getLocStart());
  SourceLocation IdLoc = Importer.Import(DTemplated->getLocation());
  TypeSourceInfo *TInfo = Importer.Import(DTemplated->getTypeSourceInfo());
  VarDecl *D2Templated = VarDecl::Create(Importer.getToContext(), DC, StartLoc,
                                         IdLoc, Name.getAsIdentifierInfo(), T,
                                         TInfo, DTemplated->getStorageClass());
  D2Templated->setAccess(DTemplated->getAccess());
  D2Templated->setQualifierInfo(Importer.Import(DTemplated->getQualifierLoc()));
  D2Templated->setLexicalDeclContext(LexicalDC);

  // Importer.Imported(DTemplated, D2Templated);
  // LexicalDC->addDeclInternal(D2Templated);

  // Merge the initializer.
  if (ImportDefinition(DTemplated, D2Templated))
    return nullptr;

  // Create the variable template declaration itself.
  TemplateParameterList *TemplateParams =
      ImportTemplateParameterList(D->getTemplateParameters());
  if (!TemplateParams)
    return nullptr;

  VarTemplateDecl *D2 = VarTemplateDecl::Create(
      Importer.getToContext(), DC, Loc, Name, TemplateParams, D2Templated);
  D2Templated->setDescribedVarTemplate(D2);

  D2->setAccess(D->getAccess());
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);

  // Note the relationship between the variable templates.
  Importer.Imported(D, D2);
  Importer.Imported(DTemplated, D2Templated);

  if (DTemplated->isThisDeclarationADefinition() &&
      !D2Templated->isThisDeclarationADefinition()) {
    // FIXME: Import definition!
  }

  return D2;
}

Decl *ASTNodeImporter::VisitVarTemplateSpecializationDecl(
    VarTemplateSpecializationDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  VarDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return nullptr;

    return Importer.Imported(D, ImportedDef);
  }

  VarTemplateDecl *VarTemplate = cast_or_null<VarTemplateDecl>(
      Importer.Import(D->getSpecializedTemplate()));
  if (!VarTemplate)
    return nullptr;

  // Import the context of this declaration.
  DeclContext *DC = VarTemplate->getDeclContext();
  if (!DC)
    return nullptr;

  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return nullptr;
  }

  // Import the location of this declaration.
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  SourceLocation IdLoc = Importer.Import(D->getLocation());

  // Import template arguments.
  SmallVector<TemplateArgument, 2> TemplateArgs;
  if (ImportTemplateArguments(D->getTemplateArgs().data(),
                              D->getTemplateArgs().size(), TemplateArgs))
    return nullptr;

  // Try to find an existing specialization with these template arguments.
  void *InsertPos = nullptr;
  VarTemplateSpecializationDecl *D2 = VarTemplate->findSpecialization(
      TemplateArgs, InsertPos);
  if (D2) {
    // We already have a variable template specialization with these template
    // arguments.

    // FIXME: Check for specialization vs. instantiation errors.

    if (VarDecl *FoundDef = D2->getDefinition()) {
      if (!D->isThisDeclarationADefinition() ||
          IsStructuralMatch(D, FoundDef)) {
        // The record types structurally match, or the "from" translation
        // unit only had a forward declaration anyway; call it the same
        // variable.
        return Importer.Imported(D, FoundDef);
      }
    }
  } else {

    // Import the type.
    QualType T = Importer.Import(D->getType());
    if (T.isNull())
      return nullptr;
    TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());

    // Create a new specialization.
    D2 = VarTemplateSpecializationDecl::Create(
        Importer.getToContext(), DC, StartLoc, IdLoc, VarTemplate, T, TInfo,
        D->getStorageClass(), TemplateArgs);
    D2->setSpecializationKind(D->getSpecializationKind());
    D2->setTemplateArgsInfo(D->getTemplateArgsInfo());

    // Add this specialization to the class template.
    VarTemplate->AddSpecialization(D2, InsertPos);

    // Import the qualifier, if any.
    D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));

    // Add the specialization to this context.
    D2->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(D2);
  }
  Importer.Imported(D, D2);

  if (D->isThisDeclarationADefinition() && ImportDefinition(D, D2))
    return nullptr;

  return D2;
}

//----------------------------------------------------------------------------
// Import Statements
//----------------------------------------------------------------------------

DeclGroupRef ASTNodeImporter::ImportDeclGroup(DeclGroupRef DG) {
  if (DG.isNull())
    return DeclGroupRef::Create(Importer.getToContext(), nullptr, 0);
  size_t NumDecls = DG.end() - DG.begin();
  SmallVector<Decl *, 1> ToDecls(NumDecls);
  auto &_Importer = this->Importer;
  std::transform(DG.begin(), DG.end(), ToDecls.begin(),
    [&_Importer](Decl *D) -> Decl * {
      return _Importer.Import(D);
    });
  return DeclGroupRef::Create(Importer.getToContext(),
                              ToDecls.begin(),
                              NumDecls);
}

 Stmt *ASTNodeImporter::VisitStmt(Stmt *S) {
   Importer.FromDiag(S->getLocStart(), diag::err_unsupported_ast_node)
     << S->getStmtClassName();
   return nullptr;
 }


Stmt *ASTNodeImporter::VisitGCCAsmStmt(GCCAsmStmt *S) {
  SmallVector<IdentifierInfo *, 4> Names;
  for (unsigned I = 0, E = S->getNumOutputs(); I != E; I++) {
    IdentifierInfo *ToII = Importer.Import(S->getOutputIdentifier(I));
    // ToII is nullptr when no symbolic name is given for output operand
    // see ParseStmtAsm::ParseAsmOperandsOpt
    if (!ToII && S->getOutputIdentifier(I))
      return nullptr;
    Names.push_back(ToII);
  }
  for (unsigned I = 0, E = S->getNumInputs(); I != E; I++) {
    IdentifierInfo *ToII = Importer.Import(S->getInputIdentifier(I));
    // ToII is nullptr when no symbolic name is given for input operand
    // see ParseStmtAsm::ParseAsmOperandsOpt
    if (!ToII && S->getInputIdentifier(I))
      return nullptr;
    Names.push_back(ToII);
  }

  SmallVector<StringLiteral *, 4> Clobbers;
  for (unsigned I = 0, E = S->getNumClobbers(); I != E; I++) {
    StringLiteral *Clobber = cast_or_null<StringLiteral>(
          Importer.Import(S->getClobberStringLiteral(I)));
    if (!Clobber)
      return nullptr;
    Clobbers.push_back(Clobber);
  }

  SmallVector<StringLiteral *, 4> Constraints;
  for (unsigned I = 0, E = S->getNumOutputs(); I != E; I++) {
    StringLiteral *Output = cast_or_null<StringLiteral>(
          Importer.Import(S->getOutputConstraintLiteral(I)));
    if (!Output)
      return nullptr;
    Constraints.push_back(Output);
  }

  for (unsigned I = 0, E = S->getNumInputs(); I != E; I++) {
    StringLiteral *Input = cast_or_null<StringLiteral>(
          Importer.Import(S->getInputConstraintLiteral(I)));
    if (!Input)
      return nullptr;
    Constraints.push_back(Input);
  }

  SmallVector<Expr *, 4> Exprs(S->getNumOutputs() + S->getNumInputs());
  if (ImportContainerChecked(S->outputs(), Exprs))
    return nullptr;

  if (ImportArrayChecked(S->inputs(), Exprs.begin() + S->getNumOutputs()))
    return nullptr;

  StringLiteral *AsmStr = cast_or_null<StringLiteral>(
        Importer.Import(S->getAsmString()));
  if (!AsmStr)
    return nullptr;

  return new (Importer.getToContext()) GCCAsmStmt(
        Importer.getToContext(),
        Importer.Import(S->getAsmLoc()),
        S->isSimple(),
        S->isVolatile(),
        S->getNumOutputs(),
        S->getNumInputs(),
        Names.data(),
        Constraints.data(),
        Exprs.data(),
        AsmStr,
        S->getNumClobbers(),
        Clobbers.data(),
        Importer.Import(S->getRParenLoc()));
}

Stmt *ASTNodeImporter::VisitDeclStmt(DeclStmt *S) {
  DeclGroupRef ToDG = ImportDeclGroup(S->getDeclGroup());
  for (Decl *ToD : ToDG) {
    if (!ToD)
      return nullptr;
  }
  SourceLocation ToStartLoc = Importer.Import(S->getStartLoc());
  SourceLocation ToEndLoc = Importer.Import(S->getEndLoc());
  return new (Importer.getToContext()) DeclStmt(ToDG, ToStartLoc, ToEndLoc);
}

Stmt *ASTNodeImporter::VisitNullStmt(NullStmt *S) {
  SourceLocation ToSemiLoc = Importer.Import(S->getSemiLoc());
  return new (Importer.getToContext()) NullStmt(ToSemiLoc,
                                                S->hasLeadingEmptyMacro());
}

Stmt *ASTNodeImporter::VisitCompoundStmt(CompoundStmt *S) {
  llvm::SmallVector<Stmt *, 8> ToStmts(S->size());

  if (ImportContainerChecked(S->body(), ToStmts))
    return nullptr;

  SourceLocation ToLBraceLoc = Importer.Import(S->getLBracLoc());
  SourceLocation ToRBraceLoc = Importer.Import(S->getRBracLoc());
  return new (Importer.getToContext()) CompoundStmt(Importer.getToContext(),
                                                    ToStmts,
                                                    ToLBraceLoc, ToRBraceLoc);
}

Stmt *ASTNodeImporter::VisitCaseStmt(CaseStmt *S) {
  Expr *ToLHS = Importer.Import(S->getLHS());
  if (!ToLHS)
    return nullptr;
  Expr *ToRHS = Importer.Import(S->getRHS());
  if (!ToRHS && S->getRHS())
    return nullptr;
  SourceLocation ToCaseLoc = Importer.Import(S->getCaseLoc());
  SourceLocation ToEllipsisLoc = Importer.Import(S->getEllipsisLoc());
  SourceLocation ToColonLoc = Importer.Import(S->getColonLoc());
  return new (Importer.getToContext()) CaseStmt(ToLHS, ToRHS,
                                                ToCaseLoc, ToEllipsisLoc,
                                                ToColonLoc);
}

Stmt *ASTNodeImporter::VisitDefaultStmt(DefaultStmt *S) {
  SourceLocation ToDefaultLoc = Importer.Import(S->getDefaultLoc());
  SourceLocation ToColonLoc = Importer.Import(S->getColonLoc());
  Stmt *ToSubStmt = Importer.Import(S->getSubStmt());
  if (!ToSubStmt && S->getSubStmt())
    return nullptr;
  return new (Importer.getToContext()) DefaultStmt(ToDefaultLoc, ToColonLoc,
                                                   ToSubStmt);
}

Stmt *ASTNodeImporter::VisitLabelStmt(LabelStmt *S) {
  SourceLocation ToIdentLoc = Importer.Import(S->getIdentLoc());
  LabelDecl *ToLabelDecl =
    cast_or_null<LabelDecl>(Importer.Import(S->getDecl()));
  if (!ToLabelDecl && S->getDecl())
    return nullptr;
  Stmt *ToSubStmt = Importer.Import(S->getSubStmt());
  if (!ToSubStmt && S->getSubStmt())
    return nullptr;
  return new (Importer.getToContext()) LabelStmt(ToIdentLoc, ToLabelDecl,
                                                 ToSubStmt);
}

Stmt *ASTNodeImporter::VisitAttributedStmt(AttributedStmt *S) {
  SourceLocation ToAttrLoc = Importer.Import(S->getAttrLoc());
  ArrayRef<const Attr*> FromAttrs(S->getAttrs());
  SmallVector<const Attr *, 1> ToAttrs(FromAttrs.size());
  ASTContext &_ToContext = Importer.getToContext();
  std::transform(FromAttrs.begin(), FromAttrs.end(), ToAttrs.begin(),
    [&_ToContext](const Attr *A) -> const Attr * {
      return A->clone(_ToContext);
    });
  for (const Attr *ToA : ToAttrs) {
    if (!ToA)
      return nullptr;
  }
  Stmt *ToSubStmt = Importer.Import(S->getSubStmt());
  if (!ToSubStmt && S->getSubStmt())
    return nullptr;
  return AttributedStmt::Create(Importer.getToContext(), ToAttrLoc,
                                ToAttrs, ToSubStmt);
}

Stmt *ASTNodeImporter::VisitIfStmt(IfStmt *S) {
  SourceLocation ToIfLoc = Importer.Import(S->getIfLoc());
  Stmt *ToInit = Importer.Import(S->getInit());
  if (!ToInit && S->getInit())
    return nullptr;
  VarDecl *ToConditionVariable = nullptr;
  if (VarDecl *FromConditionVariable = S->getConditionVariable()) {
    ToConditionVariable =
      dyn_cast_or_null<VarDecl>(Importer.Import(FromConditionVariable));
    if (!ToConditionVariable)
      return nullptr;
  }
  Expr *ToCondition = Importer.Import(S->getCond());
  if (!ToCondition && S->getCond())
    return nullptr;
  Stmt *ToThenStmt = Importer.Import(S->getThen());
  if (!ToThenStmt && S->getThen())
    return nullptr;
  SourceLocation ToElseLoc = Importer.Import(S->getElseLoc());
  Stmt *ToElseStmt = Importer.Import(S->getElse());
  if (!ToElseStmt && S->getElse())
    return nullptr;
  return new (Importer.getToContext()) IfStmt(Importer.getToContext(),
                                              ToIfLoc, S->isConstexpr(),
                                              ToInit,
                                              ToConditionVariable,
                                              ToCondition, ToThenStmt,
                                              ToElseLoc, ToElseStmt);
}

Stmt *ASTNodeImporter::VisitSwitchStmt(SwitchStmt *S) {
  Stmt *ToInit = Importer.Import(S->getInit());
  if (!ToInit && S->getInit())
    return nullptr;
  VarDecl *ToConditionVariable = nullptr;
  if (VarDecl *FromConditionVariable = S->getConditionVariable()) {
    ToConditionVariable =
      dyn_cast_or_null<VarDecl>(Importer.Import(FromConditionVariable));
    if (!ToConditionVariable)
      return nullptr;
  }
  Expr *ToCondition = Importer.Import(S->getCond());
  if (!ToCondition && S->getCond())
    return nullptr;
  SwitchStmt *ToStmt = new (Importer.getToContext()) SwitchStmt(
                         Importer.getToContext(), ToInit,
                         ToConditionVariable, ToCondition);
  Stmt *ToBody = Importer.Import(S->getBody());
  if (!ToBody && S->getBody())
    return nullptr;
  ToStmt->setBody(ToBody);
  ToStmt->setSwitchLoc(Importer.Import(S->getSwitchLoc()));
  // Now we have to re-chain the cases.
  SwitchCase *LastChainedSwitchCase = nullptr;
  for (SwitchCase *SC = S->getSwitchCaseList(); SC != nullptr;
       SC = SC->getNextSwitchCase()) {
    SwitchCase *ToSC = dyn_cast_or_null<SwitchCase>(Importer.Import(SC));
    if (!ToSC)
      return nullptr;
    if (LastChainedSwitchCase)
      LastChainedSwitchCase->setNextSwitchCase(ToSC);
    else
      ToStmt->setSwitchCaseList(ToSC);
    LastChainedSwitchCase = ToSC;
  }
  return ToStmt;
}

Stmt *ASTNodeImporter::VisitWhileStmt(WhileStmt *S) {
  VarDecl *ToConditionVariable = nullptr;
  if (VarDecl *FromConditionVariable = S->getConditionVariable()) {
    ToConditionVariable =
      dyn_cast_or_null<VarDecl>(Importer.Import(FromConditionVariable));
    if (!ToConditionVariable)
      return nullptr;
  }
  Expr *ToCondition = Importer.Import(S->getCond());
  if (!ToCondition && S->getCond())
    return nullptr;
  Stmt *ToBody = Importer.Import(S->getBody());
  if (!ToBody && S->getBody())
    return nullptr;
  SourceLocation ToWhileLoc = Importer.Import(S->getWhileLoc());
  return new (Importer.getToContext()) WhileStmt(Importer.getToContext(),
                                                 ToConditionVariable,
                                                 ToCondition, ToBody,
                                                 ToWhileLoc);
}

Stmt *ASTNodeImporter::VisitDoStmt(DoStmt *S) {
  Stmt *ToBody = Importer.Import(S->getBody());
  if (!ToBody && S->getBody())
    return nullptr;
  Expr *ToCondition = Importer.Import(S->getCond());
  if (!ToCondition && S->getCond())
    return nullptr;
  SourceLocation ToDoLoc = Importer.Import(S->getDoLoc());
  SourceLocation ToWhileLoc = Importer.Import(S->getWhileLoc());
  SourceLocation ToRParenLoc = Importer.Import(S->getRParenLoc());
  return new (Importer.getToContext()) DoStmt(ToBody, ToCondition,
                                              ToDoLoc, ToWhileLoc,
                                              ToRParenLoc);
}

Stmt *ASTNodeImporter::VisitForStmt(ForStmt *S) {
  Stmt *ToInit = Importer.Import(S->getInit());
  if (!ToInit && S->getInit())
    return nullptr;
  Expr *ToCondition = Importer.Import(S->getCond());
  if (!ToCondition && S->getCond())
    return nullptr;
  VarDecl *ToConditionVariable = nullptr;
  if (VarDecl *FromConditionVariable = S->getConditionVariable()) {
    ToConditionVariable =
      dyn_cast_or_null<VarDecl>(Importer.Import(FromConditionVariable));
    if (!ToConditionVariable)
      return nullptr;
  }
  Expr *ToInc = Importer.Import(S->getInc());
  if (!ToInc && S->getInc())
    return nullptr;
  Stmt *ToBody = Importer.Import(S->getBody());
  if (!ToBody && S->getBody())
    return nullptr;
  SourceLocation ToForLoc = Importer.Import(S->getForLoc());
  SourceLocation ToLParenLoc = Importer.Import(S->getLParenLoc());
  SourceLocation ToRParenLoc = Importer.Import(S->getRParenLoc());
  return new (Importer.getToContext()) ForStmt(Importer.getToContext(),
                                               ToInit, ToCondition,
                                               ToConditionVariable,
                                               ToInc, ToBody,
                                               ToForLoc, ToLParenLoc,
                                               ToRParenLoc);
}

Stmt *ASTNodeImporter::VisitGotoStmt(GotoStmt *S) {
  LabelDecl *ToLabel = nullptr;
  if (LabelDecl *FromLabel = S->getLabel()) {
    ToLabel = dyn_cast_or_null<LabelDecl>(Importer.Import(FromLabel));
    if (!ToLabel)
      return nullptr;
  }
  SourceLocation ToGotoLoc = Importer.Import(S->getGotoLoc());
  SourceLocation ToLabelLoc = Importer.Import(S->getLabelLoc());
  return new (Importer.getToContext()) GotoStmt(ToLabel,
                                                ToGotoLoc, ToLabelLoc);
}

Stmt *ASTNodeImporter::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  SourceLocation ToGotoLoc = Importer.Import(S->getGotoLoc());
  SourceLocation ToStarLoc = Importer.Import(S->getStarLoc());
  Expr *ToTarget = Importer.Import(S->getTarget());
  if (!ToTarget && S->getTarget())
    return nullptr;
  return new (Importer.getToContext()) IndirectGotoStmt(ToGotoLoc, ToStarLoc,
                                                        ToTarget);
}

Stmt *ASTNodeImporter::VisitContinueStmt(ContinueStmt *S) {
  SourceLocation ToContinueLoc = Importer.Import(S->getContinueLoc());
  return new (Importer.getToContext()) ContinueStmt(ToContinueLoc);
}

Stmt *ASTNodeImporter::VisitBreakStmt(BreakStmt *S) {
  SourceLocation ToBreakLoc = Importer.Import(S->getBreakLoc());
  return new (Importer.getToContext()) BreakStmt(ToBreakLoc);
}

Stmt *ASTNodeImporter::VisitReturnStmt(ReturnStmt *S) {
  SourceLocation ToRetLoc = Importer.Import(S->getReturnLoc());
  Expr *ToRetExpr = Importer.Import(S->getRetValue());
  if (!ToRetExpr && S->getRetValue())
    return nullptr;
  VarDecl *NRVOCandidate = const_cast<VarDecl*>(S->getNRVOCandidate());
  VarDecl *ToNRVOCandidate = cast_or_null<VarDecl>(Importer.Import(NRVOCandidate));
  if (!ToNRVOCandidate && NRVOCandidate)
    return nullptr;
  return new (Importer.getToContext()) ReturnStmt(ToRetLoc, ToRetExpr,
                                                  ToNRVOCandidate);
}

Stmt *ASTNodeImporter::VisitCXXCatchStmt(CXXCatchStmt *S) {
  SourceLocation ToCatchLoc = Importer.Import(S->getCatchLoc());
  VarDecl *ToExceptionDecl = nullptr;
  if (VarDecl *FromExceptionDecl = S->getExceptionDecl()) {
    ToExceptionDecl =
      dyn_cast_or_null<VarDecl>(Importer.Import(FromExceptionDecl));
    if (!ToExceptionDecl)
      return nullptr;
  }
  Stmt *ToHandlerBlock = Importer.Import(S->getHandlerBlock());
  if (!ToHandlerBlock && S->getHandlerBlock())
    return nullptr;
  return new (Importer.getToContext()) CXXCatchStmt(ToCatchLoc,
                                                    ToExceptionDecl,
                                                    ToHandlerBlock);
}

Stmt *ASTNodeImporter::VisitCXXTryStmt(CXXTryStmt *S) {
  SourceLocation ToTryLoc = Importer.Import(S->getTryLoc());
  Stmt *ToTryBlock = Importer.Import(S->getTryBlock());
  if (!ToTryBlock && S->getTryBlock())
    return nullptr;
  SmallVector<Stmt *, 1> ToHandlers(S->getNumHandlers());
  for (unsigned HI = 0, HE = S->getNumHandlers(); HI != HE; ++HI) {
    CXXCatchStmt *FromHandler = S->getHandler(HI);
    if (Stmt *ToHandler = Importer.Import(FromHandler))
      ToHandlers[HI] = ToHandler;
    else
      return nullptr;
  }
  return CXXTryStmt::Create(Importer.getToContext(), ToTryLoc, ToTryBlock,
                            ToHandlers);
}

Stmt *ASTNodeImporter::VisitCXXForRangeStmt(CXXForRangeStmt *S) {
  DeclStmt *ToRange =
    dyn_cast_or_null<DeclStmt>(Importer.Import(S->getRangeStmt()));
  if (!ToRange && S->getRangeStmt())
    return nullptr;
  DeclStmt *ToBegin =
    dyn_cast_or_null<DeclStmt>(Importer.Import(S->getBeginStmt()));
  if (!ToBegin && S->getBeginStmt())
    return nullptr;
  DeclStmt *ToEnd =
    dyn_cast_or_null<DeclStmt>(Importer.Import(S->getEndStmt()));
  if (!ToEnd && S->getEndStmt())
    return nullptr;
  Expr *ToCond = Importer.Import(S->getCond());
  if (!ToCond && S->getCond())
    return nullptr;
  Expr *ToInc = Importer.Import(S->getInc());
  if (!ToInc && S->getInc())
    return nullptr;
  DeclStmt *ToLoopVar =
    dyn_cast_or_null<DeclStmt>(Importer.Import(S->getLoopVarStmt()));
  if (!ToLoopVar && S->getLoopVarStmt())
    return nullptr;
  Stmt *ToBody = Importer.Import(S->getBody());
  if (!ToBody && S->getBody())
    return nullptr;
  SourceLocation ToForLoc = Importer.Import(S->getForLoc());
  SourceLocation ToCoawaitLoc = Importer.Import(S->getCoawaitLoc());
  SourceLocation ToColonLoc = Importer.Import(S->getColonLoc());
  SourceLocation ToRParenLoc = Importer.Import(S->getRParenLoc());
  return new (Importer.getToContext()) CXXForRangeStmt(ToRange, ToBegin, ToEnd,
                                                       ToCond, ToInc,
                                                       ToLoopVar, ToBody,
                                                       ToForLoc, ToCoawaitLoc,
                                                       ToColonLoc, ToRParenLoc);
}

Stmt *ASTNodeImporter::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  Stmt *ToElem = Importer.Import(S->getElement());
  if (!ToElem && S->getElement())
    return nullptr;
  Expr *ToCollect = Importer.Import(S->getCollection());
  if (!ToCollect && S->getCollection())
    return nullptr;
  Stmt *ToBody = Importer.Import(S->getBody());
  if (!ToBody && S->getBody())
    return nullptr;
  SourceLocation ToForLoc = Importer.Import(S->getForLoc());
  SourceLocation ToRParenLoc = Importer.Import(S->getRParenLoc());
  return new (Importer.getToContext()) ObjCForCollectionStmt(ToElem,
                                                             ToCollect,
                                                             ToBody, ToForLoc,
                                                             ToRParenLoc);
}

Stmt *ASTNodeImporter::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  SourceLocation ToAtCatchLoc = Importer.Import(S->getAtCatchLoc());
  SourceLocation ToRParenLoc = Importer.Import(S->getRParenLoc());
  VarDecl *ToExceptionDecl = nullptr;
  if (VarDecl *FromExceptionDecl = S->getCatchParamDecl()) {
    ToExceptionDecl =
      dyn_cast_or_null<VarDecl>(Importer.Import(FromExceptionDecl));
    if (!ToExceptionDecl)
      return nullptr;
  }
  Stmt *ToBody = Importer.Import(S->getCatchBody());
  if (!ToBody && S->getCatchBody())
    return nullptr;
  return new (Importer.getToContext()) ObjCAtCatchStmt(ToAtCatchLoc,
                                                       ToRParenLoc,
                                                       ToExceptionDecl,
                                                       ToBody);
}

Stmt *ASTNodeImporter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  SourceLocation ToAtFinallyLoc = Importer.Import(S->getAtFinallyLoc());
  Stmt *ToAtFinallyStmt = Importer.Import(S->getFinallyBody());
  if (!ToAtFinallyStmt && S->getFinallyBody())
    return nullptr;
  return new (Importer.getToContext()) ObjCAtFinallyStmt(ToAtFinallyLoc,
                                                         ToAtFinallyStmt);
}

Stmt *ASTNodeImporter::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  SourceLocation ToAtTryLoc = Importer.Import(S->getAtTryLoc());
  Stmt *ToAtTryStmt = Importer.Import(S->getTryBody());
  if (!ToAtTryStmt && S->getTryBody())
    return nullptr;
  SmallVector<Stmt *, 1> ToCatchStmts(S->getNumCatchStmts());
  for (unsigned CI = 0, CE = S->getNumCatchStmts(); CI != CE; ++CI) {
    ObjCAtCatchStmt *FromCatchStmt = S->getCatchStmt(CI);
    if (Stmt *ToCatchStmt = Importer.Import(FromCatchStmt))
      ToCatchStmts[CI] = ToCatchStmt;
    else
      return nullptr;
  }
  Stmt *ToAtFinallyStmt = Importer.Import(S->getFinallyStmt());
  if (!ToAtFinallyStmt && S->getFinallyStmt())
    return nullptr;
  return ObjCAtTryStmt::Create(Importer.getToContext(),
                               ToAtTryLoc, ToAtTryStmt,
                               ToCatchStmts.begin(), ToCatchStmts.size(),
                               ToAtFinallyStmt);
}

Stmt *ASTNodeImporter::VisitObjCAtSynchronizedStmt
  (ObjCAtSynchronizedStmt *S) {
  SourceLocation ToAtSynchronizedLoc =
    Importer.Import(S->getAtSynchronizedLoc());
  Expr *ToSynchExpr = Importer.Import(S->getSynchExpr());
  if (!ToSynchExpr && S->getSynchExpr())
    return nullptr;
  Stmt *ToSynchBody = Importer.Import(S->getSynchBody());
  if (!ToSynchBody && S->getSynchBody())
    return nullptr;
  return new (Importer.getToContext()) ObjCAtSynchronizedStmt(
    ToAtSynchronizedLoc, ToSynchExpr, ToSynchBody);
}

Stmt *ASTNodeImporter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  SourceLocation ToAtThrowLoc = Importer.Import(S->getThrowLoc());
  Expr *ToThrow = Importer.Import(S->getThrowExpr());
  if (!ToThrow && S->getThrowExpr())
    return nullptr;
  return new (Importer.getToContext()) ObjCAtThrowStmt(ToAtThrowLoc, ToThrow);
}

Stmt *ASTNodeImporter::VisitObjCAutoreleasePoolStmt
  (ObjCAutoreleasePoolStmt *S) {
  SourceLocation ToAtLoc = Importer.Import(S->getAtLoc());
  Stmt *ToSubStmt = Importer.Import(S->getSubStmt());
  if (!ToSubStmt && S->getSubStmt())
    return nullptr;
  return new (Importer.getToContext()) ObjCAutoreleasePoolStmt(ToAtLoc,
                                                               ToSubStmt);
}

//----------------------------------------------------------------------------
// Import Expressions
//----------------------------------------------------------------------------
Expr *ASTNodeImporter::VisitExpr(Expr *E) {
  Importer.FromDiag(E->getLocStart(), diag::err_unsupported_ast_node)
    << E->getStmtClassName();
  return nullptr;
}

Expr *ASTNodeImporter::VisitVAArgExpr(VAArgExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr && E->getSubExpr())
    return nullptr;

  TypeSourceInfo *TInfo = Importer.Import(E->getWrittenTypeInfo());
  if (!TInfo)
    return nullptr;

  return new (Importer.getToContext()) VAArgExpr(
        Importer.Import(E->getBuiltinLoc()), SubExpr, TInfo,
        Importer.Import(E->getRParenLoc()), T, E->isMicrosoftABI());
}


Expr *ASTNodeImporter::VisitGNUNullExpr(GNUNullExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  return new (Importer.getToContext()) GNUNullExpr(
        T, Importer.Import(E->getLocStart()));
}

Expr *ASTNodeImporter::VisitPredefinedExpr(PredefinedExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  StringLiteral *SL = cast_or_null<StringLiteral>(
        Importer.Import(E->getFunctionName()));
  if (!SL && E->getFunctionName())
    return nullptr;

  return new (Importer.getToContext()) PredefinedExpr(
        Importer.Import(E->getLocStart()), T, E->getIdentType(), SL);
}

Expr *ASTNodeImporter::VisitDeclRefExpr(DeclRefExpr *E) {
  ValueDecl *ToD = cast_or_null<ValueDecl>(Importer.Import(E->getDecl()));
  if (!ToD)
    return nullptr;

  NamedDecl *FoundD = nullptr;
  if (E->getDecl() != E->getFoundDecl()) {
    FoundD = cast_or_null<NamedDecl>(Importer.Import(E->getFoundDecl()));
    if (!FoundD)
      return nullptr;
  }
  
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;


  TemplateArgumentListInfo ToTAInfo;
  TemplateArgumentListInfo *ResInfo = nullptr;
  if (E->hasExplicitTemplateArgs()) {
    for (const auto &FromLoc : E->template_arguments()) {
      bool Error = false;
      TemplateArgumentLoc ToTALoc = ImportTemplateArgumentLoc(FromLoc, Error);
      if (Error)
        return nullptr;
      ToTAInfo.addArgument(ToTALoc);
    }
    ResInfo = &ToTAInfo;
  }

  DeclRefExpr *DRE = DeclRefExpr::Create(Importer.getToContext(), 
                                         Importer.Import(E->getQualifierLoc()),
                                   Importer.Import(E->getTemplateKeywordLoc()),
                                         ToD,
                                        E->refersToEnclosingVariableOrCapture(),
                                         Importer.Import(E->getLocation()),
                                         T, E->getValueKind(),
                                         FoundD, ResInfo);
  if (E->hadMultipleCandidates())
    DRE->setHadMultipleCandidates(true);
  return DRE;
}

Expr *ASTNodeImporter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  return new (Importer.getToContext()) ImplicitValueInitExpr(T);
}

ASTNodeImporter::Designator
ASTNodeImporter::ImportDesignator(const Designator &D) {
  if (D.isFieldDesignator()) {
    IdentifierInfo *ToFieldName = Importer.Import(D.getFieldName());
    // Caller checks for import error
    return Designator(ToFieldName, Importer.Import(D.getDotLoc()),
                      Importer.Import(D.getFieldLoc()));
  }
  if (D.isArrayDesignator())
    return Designator(D.getFirstExprIndex(),
                      Importer.Import(D.getLBracketLoc()),
                      Importer.Import(D.getRBracketLoc()));

  assert(D.isArrayRangeDesignator());
  return Designator(D.getFirstExprIndex(),
                    Importer.Import(D.getLBracketLoc()),
                    Importer.Import(D.getEllipsisLoc()),
                    Importer.Import(D.getRBracketLoc()));
}


Expr *ASTNodeImporter::VisitDesignatedInitExpr(DesignatedInitExpr *DIE) {
  Expr *Init = cast_or_null<Expr>(Importer.Import(DIE->getInit()));
  if (!Init)
    return nullptr;

  SmallVector<Expr *, 4> IndexExprs(DIE->getNumSubExprs() - 1);
  // List elements from the second, the first is Init itself
  for (unsigned I = 1, E = DIE->getNumSubExprs(); I < E; I++) {
    if (Expr *Arg = cast_or_null<Expr>(Importer.Import(DIE->getSubExpr(I))))
      IndexExprs[I - 1] = Arg;
    else
      return nullptr;
  }

  SmallVector<Designator, 4> Designators(DIE->size());
  llvm::transform(DIE->designators(), Designators.begin(),
                  [this](const Designator &D) -> Designator {
                    return ImportDesignator(D);
                  });

  for (const Designator &D : DIE->designators())
    if (D.isFieldDesignator() && !D.getFieldName())
      return nullptr;

  return DesignatedInitExpr::Create(
        Importer.getToContext(), Designators,
        IndexExprs, Importer.Import(DIE->getEqualOrColonLoc()),
        DIE->usesGNUSyntax(), Init);
}

Expr *ASTNodeImporter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  return new (Importer.getToContext())
      CXXNullPtrLiteralExpr(T, Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitIntegerLiteral(IntegerLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  return IntegerLiteral::Create(Importer.getToContext(), 
                                E->getValue(), T,
                                Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitFloatingLiteral(FloatingLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  return FloatingLiteral::Create(Importer.getToContext(),
                                E->getValue(), E->isExact(), T,
                                Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitCharacterLiteral(CharacterLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  return new (Importer.getToContext()) CharacterLiteral(E->getValue(),
                                                        E->getKind(), T,
                                          Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitStringLiteral(StringLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  SmallVector<SourceLocation, 4> Locations(E->getNumConcatenated());
  ImportArray(E->tokloc_begin(), E->tokloc_end(), Locations.begin());

  return StringLiteral::Create(Importer.getToContext(), E->getBytes(),
                               E->getKind(), E->isPascal(), T,
                               Locations.data(), Locations.size());
}

Expr *ASTNodeImporter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  TypeSourceInfo *TInfo = Importer.Import(E->getTypeSourceInfo());
  if (!TInfo)
    return nullptr;

  Expr *Init = Importer.Import(E->getInitializer());
  if (!Init)
    return nullptr;

  return new (Importer.getToContext()) CompoundLiteralExpr(
        Importer.Import(E->getLParenLoc()), TInfo, T, E->getValueKind(),
        Init, E->isFileScope());
}

Expr *ASTNodeImporter::VisitAtomicExpr(AtomicExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  SmallVector<Expr *, 6> Exprs(E->getNumSubExprs());
  if (ImportArrayChecked(
        E->getSubExprs(), E->getSubExprs() + E->getNumSubExprs(),
        Exprs.begin()))
    return nullptr;

  return new (Importer.getToContext()) AtomicExpr(
        Importer.Import(E->getBuiltinLoc()), Exprs, T, E->getOp(),
        Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitAddrLabelExpr(AddrLabelExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  LabelDecl *ToLabel = cast_or_null<LabelDecl>(Importer.Import(E->getLabel()));
  if (!ToLabel)
    return nullptr;

  return new (Importer.getToContext()) AddrLabelExpr(
        Importer.Import(E->getAmpAmpLoc()), Importer.Import(E->getLabelLoc()),
        ToLabel, T);
}

Expr *ASTNodeImporter::VisitParenExpr(ParenExpr *E) {
  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return nullptr;

  return new (Importer.getToContext()) 
                                  ParenExpr(Importer.Import(E->getLParen()),
                                            Importer.Import(E->getRParen()),
                                            SubExpr);
}

Expr *ASTNodeImporter::VisitParenListExpr(ParenListExpr *E) {
  SmallVector<Expr *, 4> Exprs(E->getNumExprs());
  if (ImportContainerChecked(E->exprs(), Exprs))
    return nullptr;

  return new (Importer.getToContext()) ParenListExpr(
        Importer.getToContext(), Importer.Import(E->getLParenLoc()),
        Exprs, Importer.Import(E->getLParenLoc()));
}

Expr *ASTNodeImporter::VisitStmtExpr(StmtExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  CompoundStmt *ToSubStmt = cast_or_null<CompoundStmt>(
        Importer.Import(E->getSubStmt()));
  if (!ToSubStmt && E->getSubStmt())
    return nullptr;

  return new (Importer.getToContext()) StmtExpr(ToSubStmt, T,
        Importer.Import(E->getLParenLoc()), Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitUnaryOperator(UnaryOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return nullptr;

  return new (Importer.getToContext()) UnaryOperator(SubExpr, E->getOpcode(),
                                                     T, E->getValueKind(),
                                                     E->getObjectKind(),
                                         Importer.Import(E->getOperatorLoc()));                                        
}

Expr *ASTNodeImporter::VisitUnaryExprOrTypeTraitExpr(
                                            UnaryExprOrTypeTraitExpr *E) {
  QualType ResultType = Importer.Import(E->getType());
  
  if (E->isArgumentType()) {
    TypeSourceInfo *TInfo = Importer.Import(E->getArgumentTypeInfo());
    if (!TInfo)
      return nullptr;

    return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(E->getKind(),
                                           TInfo, ResultType,
                                           Importer.Import(E->getOperatorLoc()),
                                           Importer.Import(E->getRParenLoc()));
  }
  
  Expr *SubExpr = Importer.Import(E->getArgumentExpr());
  if (!SubExpr)
    return nullptr;

  return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(E->getKind(),
                                          SubExpr, ResultType,
                                          Importer.Import(E->getOperatorLoc()),
                                          Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitBinaryOperator(BinaryOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *LHS = Importer.Import(E->getLHS());
  if (!LHS)
    return nullptr;

  Expr *RHS = Importer.Import(E->getRHS());
  if (!RHS)
    return nullptr;

  return new (Importer.getToContext()) BinaryOperator(LHS, RHS, E->getOpcode(),
                                                      T, E->getValueKind(),
                                                      E->getObjectKind(),
                                           Importer.Import(E->getOperatorLoc()),
                                                      E->getFPFeatures());
}

Expr *ASTNodeImporter::VisitConditionalOperator(ConditionalOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *ToLHS = Importer.Import(E->getLHS());
  if (!ToLHS)
    return nullptr;

  Expr *ToRHS = Importer.Import(E->getRHS());
  if (!ToRHS)
    return nullptr;

  Expr *ToCond = Importer.Import(E->getCond());
  if (!ToCond)
    return nullptr;

  return new (Importer.getToContext()) ConditionalOperator(
        ToCond, Importer.Import(E->getQuestionLoc()),
        ToLHS, Importer.Import(E->getColonLoc()),
        ToRHS, T, E->getValueKind(), E->getObjectKind());
}

Expr *ASTNodeImporter::VisitBinaryConditionalOperator(
    BinaryConditionalOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *Common = Importer.Import(E->getCommon());
  if (!Common)
    return nullptr;

  Expr *Cond = Importer.Import(E->getCond());
  if (!Cond)
    return nullptr;

  OpaqueValueExpr *OpaqueValue = cast_or_null<OpaqueValueExpr>(
        Importer.Import(E->getOpaqueValue()));
  if (!OpaqueValue)
    return nullptr;

  Expr *TrueExpr = Importer.Import(E->getTrueExpr());
  if (!TrueExpr)
    return nullptr;

  Expr *FalseExpr = Importer.Import(E->getFalseExpr());
  if (!FalseExpr)
    return nullptr;

  return new (Importer.getToContext()) BinaryConditionalOperator(
        Common, OpaqueValue, Cond, TrueExpr, FalseExpr,
        Importer.Import(E->getQuestionLoc()), Importer.Import(E->getColonLoc()),
        T, E->getValueKind(), E->getObjectKind());
}

Expr *ASTNodeImporter::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  TypeSourceInfo *ToQueried = Importer.Import(E->getQueriedTypeSourceInfo());
  if (!ToQueried)
    return nullptr;

  Expr *Dim = Importer.Import(E->getDimensionExpression());
  if (!Dim && E->getDimensionExpression())
    return nullptr;

  return new (Importer.getToContext()) ArrayTypeTraitExpr(
        Importer.Import(E->getLocStart()), E->getTrait(), ToQueried,
        E->getValue(), Dim, Importer.Import(E->getLocEnd()), T);
}

Expr *ASTNodeImporter::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *ToQueried = Importer.Import(E->getQueriedExpression());
  if (!ToQueried)
    return nullptr;

  return new (Importer.getToContext()) ExpressionTraitExpr(
        Importer.Import(E->getLocStart()), E->getTrait(), ToQueried,
        E->getValue(), Importer.Import(E->getLocEnd()), T);
}

Expr *ASTNodeImporter::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *SourceExpr = Importer.Import(E->getSourceExpr());
  if (!SourceExpr && E->getSourceExpr())
    return nullptr;

  return new (Importer.getToContext()) OpaqueValueExpr(
        Importer.Import(E->getLocation()), T, E->getValueKind(),
        E->getObjectKind(), SourceExpr);
}

Expr *ASTNodeImporter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *ToLHS = Importer.Import(E->getLHS());
  if (!ToLHS)
    return nullptr;

  Expr *ToRHS = Importer.Import(E->getRHS());
  if (!ToRHS)
    return nullptr;

  return new (Importer.getToContext()) ArraySubscriptExpr(
        ToLHS, ToRHS, T, E->getValueKind(), E->getObjectKind(),
        Importer.Import(E->getRBracketLoc()));
}

Expr *ASTNodeImporter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  QualType CompLHSType = Importer.Import(E->getComputationLHSType());
  if (CompLHSType.isNull())
    return nullptr;

  QualType CompResultType = Importer.Import(E->getComputationResultType());
  if (CompResultType.isNull())
    return nullptr;

  Expr *LHS = Importer.Import(E->getLHS());
  if (!LHS)
    return nullptr;

  Expr *RHS = Importer.Import(E->getRHS());
  if (!RHS)
    return nullptr;

  return new (Importer.getToContext()) 
                        CompoundAssignOperator(LHS, RHS, E->getOpcode(),
                                               T, E->getValueKind(),
                                               E->getObjectKind(),
                                               CompLHSType, CompResultType,
                                           Importer.Import(E->getOperatorLoc()),
                                               E->getFPFeatures());
}

bool ASTNodeImporter::ImportCastPath(CastExpr *CE, CXXCastPath &Path) {
  for (auto I = CE->path_begin(), E = CE->path_end(); I != E; ++I) {
    if (CXXBaseSpecifier *Spec = Importer.Import(*I))
      Path.push_back(Spec);
    else
      return true;
  }
  return false;
}

Expr *ASTNodeImporter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return nullptr;

  CXXCastPath BasePath;
  if (ImportCastPath(E, BasePath))
    return nullptr;

  return ImplicitCastExpr::Create(Importer.getToContext(), T, E->getCastKind(),
                                  SubExpr, &BasePath, E->getValueKind());
}

Expr *ASTNodeImporter::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return nullptr;

  TypeSourceInfo *TInfo = Importer.Import(E->getTypeInfoAsWritten());
  if (!TInfo && E->getTypeInfoAsWritten())
    return nullptr;

  CXXCastPath BasePath;
  if (ImportCastPath(E, BasePath))
    return nullptr;

  switch (E->getStmtClass()) {
  case Stmt::CStyleCastExprClass: {
    CStyleCastExpr *CCE = cast<CStyleCastExpr>(E);
    return CStyleCastExpr::Create(Importer.getToContext(), T,
                                  E->getValueKind(), E->getCastKind(),
                                  SubExpr, &BasePath, TInfo,
                                  Importer.Import(CCE->getLParenLoc()),
                                  Importer.Import(CCE->getRParenLoc()));
  }

  case Stmt::CXXFunctionalCastExprClass: {
    CXXFunctionalCastExpr *FCE = cast<CXXFunctionalCastExpr>(E);
    return CXXFunctionalCastExpr::Create(Importer.getToContext(), T,
                                         E->getValueKind(), TInfo,
                                         E->getCastKind(), SubExpr, &BasePath,
                                         Importer.Import(FCE->getLParenLoc()),
                                         Importer.Import(FCE->getRParenLoc()));
  }

  case Stmt::ObjCBridgedCastExprClass: {
      ObjCBridgedCastExpr *OCE = cast<ObjCBridgedCastExpr>(E);
      return new (Importer.getToContext()) ObjCBridgedCastExpr(
            Importer.Import(OCE->getLParenLoc()), OCE->getBridgeKind(),
            E->getCastKind(), Importer.Import(OCE->getBridgeKeywordLoc()),
            TInfo, SubExpr);
  }
  default:
    break; // just fall through
  }

  CXXNamedCastExpr *Named = cast<CXXNamedCastExpr>(E);
  SourceLocation ExprLoc = Importer.Import(Named->getOperatorLoc()),
      RParenLoc = Importer.Import(Named->getRParenLoc());
  SourceRange Brackets = Importer.Import(Named->getAngleBrackets());

  switch (E->getStmtClass()) {
  case Stmt::CXXStaticCastExprClass:
    return CXXStaticCastExpr::Create(Importer.getToContext(), T,
                                     E->getValueKind(), E->getCastKind(),
                                     SubExpr, &BasePath, TInfo,
                                     ExprLoc, RParenLoc, Brackets);

  case Stmt::CXXDynamicCastExprClass:
    return CXXDynamicCastExpr::Create(Importer.getToContext(), T,
                                      E->getValueKind(), E->getCastKind(),
                                      SubExpr, &BasePath, TInfo,
                                      ExprLoc, RParenLoc, Brackets);

  case Stmt::CXXReinterpretCastExprClass:
    return CXXReinterpretCastExpr::Create(Importer.getToContext(), T,
                                          E->getValueKind(), E->getCastKind(),
                                          SubExpr, &BasePath, TInfo,
                                          ExprLoc, RParenLoc, Brackets);

  case Stmt::CXXConstCastExprClass:
    return CXXConstCastExpr::Create(Importer.getToContext(), T,
                                    E->getValueKind(), SubExpr, TInfo, ExprLoc,
                                    RParenLoc, Brackets);
  default:
    llvm_unreachable("Cast expression of unsupported type!");
    return nullptr;
  }
}

Expr *ASTNodeImporter::VisitOffsetOfExpr(OffsetOfExpr *OE) {
  QualType T = Importer.Import(OE->getType());
  if (T.isNull())
    return nullptr;

  SmallVector<OffsetOfNode, 4> Nodes;
  for (int I = 0, E = OE->getNumComponents(); I < E; ++I) {
    const OffsetOfNode &Node = OE->getComponent(I);

    switch (Node.getKind()) {
    case OffsetOfNode::Array:
      Nodes.push_back(OffsetOfNode(Importer.Import(Node.getLocStart()),
                                   Node.getArrayExprIndex(),
                                   Importer.Import(Node.getLocEnd())));
      break;

    case OffsetOfNode::Base: {
      CXXBaseSpecifier *BS = Importer.Import(Node.getBase());
      if (!BS && Node.getBase())
        return nullptr;
      Nodes.push_back(OffsetOfNode(BS));
      break;
    }
    case OffsetOfNode::Field: {
      FieldDecl *FD = cast_or_null<FieldDecl>(Importer.Import(Node.getField()));
      if (!FD)
        return nullptr;
      Nodes.push_back(OffsetOfNode(Importer.Import(Node.getLocStart()), FD,
                                   Importer.Import(Node.getLocEnd())));
      break;
    }
    case OffsetOfNode::Identifier: {
      IdentifierInfo *ToII = Importer.Import(Node.getFieldName());
      if (!ToII)
        return nullptr;
      Nodes.push_back(OffsetOfNode(Importer.Import(Node.getLocStart()), ToII,
                                   Importer.Import(Node.getLocEnd())));
      break;
    }
    }
  }

  SmallVector<Expr *, 4> Exprs(OE->getNumExpressions());
  for (int I = 0, E = OE->getNumExpressions(); I < E; ++I) {
    Expr *ToIndexExpr = Importer.Import(OE->getIndexExpr(I));
    if (!ToIndexExpr)
      return nullptr;
    Exprs[I] = ToIndexExpr;
  }

  TypeSourceInfo *TInfo = Importer.Import(OE->getTypeSourceInfo());
  if (!TInfo && OE->getTypeSourceInfo())
    return nullptr;

  return OffsetOfExpr::Create(Importer.getToContext(), T,
                              Importer.Import(OE->getOperatorLoc()),
                              TInfo, Nodes, Exprs,
                              Importer.Import(OE->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *Operand = Importer.Import(E->getOperand());
  if (!Operand)
    return nullptr;

  CanThrowResult CanThrow;
  if (E->isValueDependent())
    CanThrow = CT_Dependent;
  else
    CanThrow = E->getValue() ? CT_Can : CT_Cannot;

  return new (Importer.getToContext()) CXXNoexceptExpr(
        T, Operand, CanThrow,
        Importer.Import(E->getLocStart()), Importer.Import(E->getLocEnd()));
}

Expr *ASTNodeImporter::VisitCXXThrowExpr(CXXThrowExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr && E->getSubExpr())
    return nullptr;

  return new (Importer.getToContext()) CXXThrowExpr(
        SubExpr, T, Importer.Import(E->getThrowLoc()),
        E->isThrownVariableInScope());
}

Expr *ASTNodeImporter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  ParmVarDecl *Param = cast_or_null<ParmVarDecl>(
        Importer.Import(E->getParam()));
  if (!Param)
    return nullptr;

  return CXXDefaultArgExpr::Create(
        Importer.getToContext(), Importer.Import(E->getUsedLocation()), Param);
}

Expr *ASTNodeImporter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  TypeSourceInfo *TypeInfo = Importer.Import(E->getTypeSourceInfo());
  if (!TypeInfo)
    return nullptr;

  return new (Importer.getToContext()) CXXScalarValueInitExpr(
        T, TypeInfo, Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return nullptr;

  auto *Dtor = cast_or_null<CXXDestructorDecl>(
        Importer.Import(const_cast<CXXDestructorDecl *>(
                          E->getTemporary()->getDestructor())));
  if (!Dtor)
    return nullptr;

  ASTContext &ToCtx = Importer.getToContext();
  CXXTemporary *Temp = CXXTemporary::Create(ToCtx, Dtor);
  return CXXBindTemporaryExpr::Create(ToCtx, Temp, SubExpr);
}

Expr *ASTNodeImporter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *CE) {
  QualType T = Importer.Import(CE->getType());
  if (T.isNull())
    return nullptr;

  SmallVector<Expr *, 8> Args(CE->getNumArgs());
  if (ImportContainerChecked(CE->arguments(), Args))
    return nullptr;

  auto *Ctor = cast_or_null<CXXConstructorDecl>(
        Importer.Import(CE->getConstructor()));
  if (!Ctor)
    return nullptr;

  return CXXTemporaryObjectExpr::Create(
        Importer.getToContext(), T,
        Importer.Import(CE->getLocStart()),
        Ctor,
        CE->isElidable(),
        Args,
        CE->hadMultipleCandidates(),
        CE->isListInitialization(),
        CE->isStdInitListInitialization(),
        CE->requiresZeroInitialization(),
        CE->getConstructionKind(),
        Importer.Import(CE->getParenOrBraceRange()));
}

Expr *
ASTNodeImporter::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *TempE = Importer.Import(E->GetTemporaryExpr());
  if (!TempE)
    return nullptr;

  ValueDecl *ExtendedBy = cast_or_null<ValueDecl>(
        Importer.Import(const_cast<ValueDecl *>(E->getExtendingDecl())));
  if (!ExtendedBy && E->getExtendingDecl())
    return nullptr;

  auto *ToMTE =  new (Importer.getToContext()) MaterializeTemporaryExpr(
        T, TempE, E->isBoundToLvalueReference());

  // FIXME: Should ManglingNumber get numbers associated with 'to' context?
  ToMTE->setExtendingDecl(ExtendedBy, E->getManglingNumber());
  return ToMTE;
}

Expr *ASTNodeImporter::VisitCXXNewExpr(CXXNewExpr *CE) {
  QualType T = Importer.Import(CE->getType());
  if (T.isNull())
    return nullptr;

  SmallVector<Expr *, 4> PlacementArgs(CE->getNumPlacementArgs());
  if (ImportContainerChecked(CE->placement_arguments(), PlacementArgs))
    return nullptr;

  FunctionDecl *OperatorNewDecl = cast_or_null<FunctionDecl>(
        Importer.Import(CE->getOperatorNew()));
  if (!OperatorNewDecl && CE->getOperatorNew())
    return nullptr;

  FunctionDecl *OperatorDeleteDecl = cast_or_null<FunctionDecl>(
        Importer.Import(CE->getOperatorDelete()));
  if (!OperatorDeleteDecl && CE->getOperatorDelete())
    return nullptr;

  Expr *ToInit = Importer.Import(CE->getInitializer());
  if (!ToInit && CE->getInitializer())
    return nullptr;

  TypeSourceInfo *TInfo = Importer.Import(CE->getAllocatedTypeSourceInfo());
  if (!TInfo)
    return nullptr;

  Expr *ToArrSize = Importer.Import(CE->getArraySize());
  if (!ToArrSize && CE->getArraySize())
    return nullptr;

  return new (Importer.getToContext()) CXXNewExpr(
        Importer.getToContext(),
        CE->isGlobalNew(),
        OperatorNewDecl, OperatorDeleteDecl,
        CE->passAlignment(),
        CE->doesUsualArrayDeleteWantSize(),
        PlacementArgs,
        Importer.Import(CE->getTypeIdParens()),
        ToArrSize, CE->getInitializationStyle(), ToInit, T, TInfo,
        Importer.Import(CE->getSourceRange()),
        Importer.Import(CE->getDirectInitRange()));
}

Expr *ASTNodeImporter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  FunctionDecl *OperatorDeleteDecl = cast_or_null<FunctionDecl>(
        Importer.Import(E->getOperatorDelete()));
  if (!OperatorDeleteDecl && E->getOperatorDelete())
    return nullptr;

  Expr *ToArg = Importer.Import(E->getArgument());
  if (!ToArg && E->getArgument())
    return nullptr;

  return new (Importer.getToContext()) CXXDeleteExpr(
        T, E->isGlobalDelete(),
        E->isArrayForm(),
        E->isArrayFormAsWritten(),
        E->doesUsualArrayDeleteWantSize(),
        OperatorDeleteDecl,
        ToArg,
        Importer.Import(E->getLocStart()));
}

Expr *ASTNodeImporter::VisitCXXConstructExpr(CXXConstructExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  CXXConstructorDecl *ToCCD =
    dyn_cast_or_null<CXXConstructorDecl>(Importer.Import(E->getConstructor()));
  if (!ToCCD)
    return nullptr;

  SmallVector<Expr *, 6> ToArgs(E->getNumArgs());
  if (ImportContainerChecked(E->arguments(), ToArgs))
    return nullptr;

  return CXXConstructExpr::Create(Importer.getToContext(), T,
                                  Importer.Import(E->getLocation()),
                                  ToCCD, E->isElidable(),
                                  ToArgs, E->hadMultipleCandidates(),
                                  E->isListInitialization(),
                                  E->isStdInitListInitialization(),
                                  E->requiresZeroInitialization(),
                                  E->getConstructionKind(),
                                  Importer.Import(E->getParenOrBraceRange()));
}

Expr *ASTNodeImporter::VisitExprWithCleanups(ExprWithCleanups *EWC) {
  Expr *SubExpr = Importer.Import(EWC->getSubExpr());
  if (!SubExpr && EWC->getSubExpr())
    return nullptr;

  SmallVector<ExprWithCleanups::CleanupObject, 8> Objs(EWC->getNumObjects());
  for (unsigned I = 0, E = EWC->getNumObjects(); I < E; I++)
    if (ExprWithCleanups::CleanupObject Obj =
        cast_or_null<BlockDecl>(Importer.Import(EWC->getObject(I))))
      Objs[I] = Obj;
    else
      return nullptr;

  return ExprWithCleanups::Create(Importer.getToContext(),
                                  SubExpr, EWC->cleanupsHaveSideEffects(),
                                  Objs);
}

Expr *ASTNodeImporter::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;
  
  Expr *ToFn = Importer.Import(E->getCallee());
  if (!ToFn)
    return nullptr;
  
  SmallVector<Expr *, 4> ToArgs(E->getNumArgs());
  if (ImportContainerChecked(E->arguments(), ToArgs))
    return nullptr;

  return new (Importer.getToContext()) CXXMemberCallExpr(
        Importer.getToContext(), ToFn, ToArgs, T, E->getValueKind(),
        Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitCXXThisExpr(CXXThisExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;
  
  return new (Importer.getToContext())
  CXXThisExpr(Importer.Import(E->getLocation()), T, E->isImplicit());
}

Expr *ASTNodeImporter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;
  
  return new (Importer.getToContext())
  CXXBoolLiteralExpr(E->getValue(), T, Importer.Import(E->getLocation()));
}


Expr *ASTNodeImporter::VisitMemberExpr(MemberExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *ToBase = Importer.Import(E->getBase());
  if (!ToBase && E->getBase())
    return nullptr;

  ValueDecl *ToMember = dyn_cast<ValueDecl>(Importer.Import(E->getMemberDecl()));
  if (!ToMember && E->getMemberDecl())
    return nullptr;

  DeclAccessPair ToFoundDecl = DeclAccessPair::make(
    dyn_cast<NamedDecl>(Importer.Import(E->getFoundDecl().getDecl())),
    E->getFoundDecl().getAccess());

  DeclarationNameInfo ToMemberNameInfo(
    Importer.Import(E->getMemberNameInfo().getName()),
    Importer.Import(E->getMemberNameInfo().getLoc()));

  if (E->hasExplicitTemplateArgs()) {
    return nullptr; // FIXME: handle template arguments
  }

  return MemberExpr::Create(Importer.getToContext(), ToBase,
                            E->isArrow(),
                            Importer.Import(E->getOperatorLoc()),
                            Importer.Import(E->getQualifierLoc()),
                            Importer.Import(E->getTemplateKeywordLoc()),
                            ToMember, ToFoundDecl, ToMemberNameInfo,
                            nullptr, T, E->getValueKind(),
                            E->getObjectKind());
}

Expr *ASTNodeImporter::VisitCallExpr(CallExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  Expr *ToCallee = Importer.Import(E->getCallee());
  if (!ToCallee && E->getCallee())
    return nullptr;

  unsigned NumArgs = E->getNumArgs();

  llvm::SmallVector<Expr *, 2> ToArgs(NumArgs);

  for (unsigned ai = 0, ae = NumArgs; ai != ae; ++ai) {
    Expr *FromArg = E->getArg(ai);
    Expr *ToArg = Importer.Import(FromArg);
    if (!ToArg)
      return nullptr;
    ToArgs[ai] = ToArg;
  }

  Expr **ToArgs_Copied = new (Importer.getToContext()) 
    Expr*[NumArgs];

  for (unsigned ai = 0, ae = NumArgs; ai != ae; ++ai)
    ToArgs_Copied[ai] = ToArgs[ai];

  return new (Importer.getToContext())
    CallExpr(Importer.getToContext(), ToCallee, 
             llvm::makeArrayRef(ToArgs_Copied, NumArgs), T, E->getValueKind(),
             Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitInitListExpr(InitListExpr *ILE) {
  QualType T = Importer.Import(ILE->getType());
  if (T.isNull())
    return nullptr;

  llvm::SmallVector<Expr *, 4> Exprs(ILE->getNumInits());
  if (ImportContainerChecked(ILE->inits(), Exprs))
    return nullptr;

  ASTContext &ToCtx = Importer.getToContext();
  InitListExpr *To = new (ToCtx) InitListExpr(
        ToCtx, Importer.Import(ILE->getLBraceLoc()),
        Exprs, Importer.Import(ILE->getLBraceLoc()));
  To->setType(T);

  if (ILE->hasArrayFiller()) {
    Expr *Filler = Importer.Import(ILE->getArrayFiller());
    if (!Filler)
      return nullptr;
    To->setArrayFiller(Filler);
  }

  if (FieldDecl *FromFD = ILE->getInitializedFieldInUnion()) {
    FieldDecl *ToFD = cast_or_null<FieldDecl>(Importer.Import(FromFD));
    if (!ToFD)
      return nullptr;
    To->setInitializedFieldInUnion(ToFD);
  }

  if (InitListExpr *SyntForm = ILE->getSyntacticForm()) {
    InitListExpr *ToSyntForm = cast_or_null<InitListExpr>(
          Importer.Import(SyntForm));
    if (!ToSyntForm)
      return nullptr;
    To->setSyntacticForm(ToSyntForm);
  }

  To->sawArrayRangeDesignator(ILE->hadArrayRangeDesignator());
  To->setValueDependent(ILE->isValueDependent());
  To->setInstantiationDependent(ILE->isInstantiationDependent());

  return To;
}

Expr *ASTNodeImporter::VisitArrayInitLoopExpr(ArrayInitLoopExpr *E) {
  QualType ToType = Importer.Import(E->getType());
  if (ToType.isNull())
    return nullptr;

  Expr *ToCommon = Importer.Import(E->getCommonExpr());
  if (!ToCommon && E->getCommonExpr())
    return nullptr;

  Expr *ToSubExpr = Importer.Import(E->getSubExpr());
  if (!ToSubExpr && E->getSubExpr())
    return nullptr;

  return new (Importer.getToContext())
      ArrayInitLoopExpr(ToType, ToCommon, ToSubExpr);
}

Expr *ASTNodeImporter::VisitArrayInitIndexExpr(ArrayInitIndexExpr *E) {
  QualType ToType = Importer.Import(E->getType());
  if (ToType.isNull())
    return nullptr;
  return new (Importer.getToContext()) ArrayInitIndexExpr(ToType);
}

Expr *ASTNodeImporter::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
  FieldDecl *ToField = llvm::dyn_cast_or_null<FieldDecl>(
      Importer.Import(DIE->getField()));
  if (!ToField && DIE->getField())
    return nullptr;

  return CXXDefaultInitExpr::Create(
      Importer.getToContext(), Importer.Import(DIE->getLocStart()), ToField);
}

Expr *ASTNodeImporter::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  QualType ToType = Importer.Import(E->getType());
  if (ToType.isNull() && !E->getType().isNull())
    return nullptr;
  ExprValueKind VK = E->getValueKind();
  CastKind CK = E->getCastKind();
  Expr *ToOp = Importer.Import(E->getSubExpr());
  if (!ToOp && E->getSubExpr())
    return nullptr;
  CXXCastPath BasePath;
  if (ImportCastPath(E, BasePath))
    return nullptr;
  TypeSourceInfo *ToWritten = Importer.Import(E->getTypeInfoAsWritten());
  SourceLocation ToOperatorLoc = Importer.Import(E->getOperatorLoc());
  SourceLocation ToRParenLoc = Importer.Import(E->getRParenLoc());
  SourceRange ToAngleBrackets = Importer.Import(E->getAngleBrackets());
  
  if (isa<CXXStaticCastExpr>(E)) {
    return CXXStaticCastExpr::Create(
        Importer.getToContext(), ToType, VK, CK, ToOp, &BasePath, 
        ToWritten, ToOperatorLoc, ToRParenLoc, ToAngleBrackets);
  } else if (isa<CXXDynamicCastExpr>(E)) {
    return CXXDynamicCastExpr::Create(
        Importer.getToContext(), ToType, VK, CK, ToOp, &BasePath, 
        ToWritten, ToOperatorLoc, ToRParenLoc, ToAngleBrackets);
  } else if (isa<CXXReinterpretCastExpr>(E)) {
    return CXXReinterpretCastExpr::Create(
        Importer.getToContext(), ToType, VK, CK, ToOp, &BasePath, 
        ToWritten, ToOperatorLoc, ToRParenLoc, ToAngleBrackets);
  } else {
    return nullptr;
  }
}


Expr *ASTNodeImporter::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return nullptr;

  NonTypeTemplateParmDecl *Param = cast_or_null<NonTypeTemplateParmDecl>(
        Importer.Import(E->getParameter()));
  if (!Param)
    return nullptr;

  Expr *Replacement = Importer.Import(E->getReplacement());
  if (!Replacement)
    return nullptr;

  return new (Importer.getToContext()) SubstNonTypeTemplateParmExpr(
        T, E->getValueKind(), Importer.Import(E->getExprLoc()), Param,
        Replacement);
}

ASTImporter::ASTImporter(ASTContext &ToContext, FileManager &ToFileManager,
                         ASTContext &FromContext, FileManager &FromFileManager,
                         bool MinimalImport)
  : ToContext(ToContext), FromContext(FromContext),
    ToFileManager(ToFileManager), FromFileManager(FromFileManager),
    Minimal(MinimalImport), LastDiagFromFrom(false)
{
  ImportedDecls[FromContext.getTranslationUnitDecl()]
    = ToContext.getTranslationUnitDecl();
}

ASTImporter::~ASTImporter() { }

QualType ASTImporter::Import(QualType FromT) {
  if (FromT.isNull())
    return QualType();

  const Type *fromTy = FromT.getTypePtr();
  
  // Check whether we've already imported this type.  
  llvm::DenseMap<const Type *, const Type *>::iterator Pos
    = ImportedTypes.find(fromTy);
  if (Pos != ImportedTypes.end())
    return ToContext.getQualifiedType(Pos->second, FromT.getLocalQualifiers());
  
  // Import the type
  ASTNodeImporter Importer(*this);
  QualType ToT = Importer.Visit(fromTy);
  if (ToT.isNull())
    return ToT;
  
  // Record the imported type.
  ImportedTypes[fromTy] = ToT.getTypePtr();
  
  return ToContext.getQualifiedType(ToT, FromT.getLocalQualifiers());
}

TypeSourceInfo *ASTImporter::Import(TypeSourceInfo *FromTSI) {
  if (!FromTSI)
    return FromTSI;

  // FIXME: For now we just create a "trivial" type source info based
  // on the type and a single location. Implement a real version of this.
  QualType T = Import(FromTSI->getType());
  if (T.isNull())
    return nullptr;

  return ToContext.getTrivialTypeSourceInfo(T, 
           Import(FromTSI->getTypeLoc().getLocStart()));
}

Decl *ASTImporter::GetAlreadyImportedOrNull(Decl *FromD) {
  llvm::DenseMap<Decl *, Decl *>::iterator Pos = ImportedDecls.find(FromD);
  if (Pos != ImportedDecls.end()) {
    Decl *ToD = Pos->second;
    ASTNodeImporter(*this).ImportDefinitionIfNeeded(FromD, ToD);
    return ToD;
  } else {
    return nullptr;
  }
}

Decl *ASTImporter::Import(Decl *FromD) {
  if (!FromD)
    return nullptr;

  ASTNodeImporter Importer(*this);

  // Check whether we've already imported this declaration.  
  llvm::DenseMap<Decl *, Decl *>::iterator Pos = ImportedDecls.find(FromD);
  if (Pos != ImportedDecls.end()) {
    Decl *ToD = Pos->second;
    Importer.ImportDefinitionIfNeeded(FromD, ToD);
    return ToD;
  }
  
  // Import the type
  Decl *ToD = Importer.Visit(FromD);
  if (!ToD)
    return nullptr;

  // Record the imported declaration.
  ImportedDecls[FromD] = ToD;
  
  if (TagDecl *FromTag = dyn_cast<TagDecl>(FromD)) {
    // Keep track of anonymous tags that have an associated typedef.
    if (FromTag->getTypedefNameForAnonDecl())
      AnonTagsWithPendingTypedefs.push_back(FromTag);
  } else if (TypedefNameDecl *FromTypedef = dyn_cast<TypedefNameDecl>(FromD)) {
    // When we've finished transforming a typedef, see whether it was the
    // typedef for an anonymous tag.
    for (SmallVectorImpl<TagDecl *>::iterator
               FromTag = AnonTagsWithPendingTypedefs.begin(), 
            FromTagEnd = AnonTagsWithPendingTypedefs.end();
         FromTag != FromTagEnd; ++FromTag) {
      if ((*FromTag)->getTypedefNameForAnonDecl() == FromTypedef) {
        if (TagDecl *ToTag = cast_or_null<TagDecl>(Import(*FromTag))) {
          // We found the typedef for an anonymous tag; link them.
          ToTag->setTypedefNameForAnonDecl(cast<TypedefNameDecl>(ToD));
          AnonTagsWithPendingTypedefs.erase(FromTag);
          break;
        }
      }
    }
  }
  
  return ToD;
}

DeclContext *ASTImporter::ImportContext(DeclContext *FromDC) {
  if (!FromDC)
    return FromDC;

  DeclContext *ToDC = cast_or_null<DeclContext>(Import(cast<Decl>(FromDC)));
  if (!ToDC)
    return nullptr;

  // When we're using a record/enum/Objective-C class/protocol as a context, we 
  // need it to have a definition.
  if (RecordDecl *ToRecord = dyn_cast<RecordDecl>(ToDC)) {
    RecordDecl *FromRecord = cast<RecordDecl>(FromDC);
    if (ToRecord->isCompleteDefinition()) {
      // Do nothing.
    } else if (FromRecord->isCompleteDefinition()) {
      ASTNodeImporter(*this).ImportDefinition(FromRecord, ToRecord,
                                              ASTNodeImporter::IDK_Basic);
    } else {
      CompleteDecl(ToRecord);
    }
  } else if (EnumDecl *ToEnum = dyn_cast<EnumDecl>(ToDC)) {
    EnumDecl *FromEnum = cast<EnumDecl>(FromDC);
    if (ToEnum->isCompleteDefinition()) {
      // Do nothing.
    } else if (FromEnum->isCompleteDefinition()) {
      ASTNodeImporter(*this).ImportDefinition(FromEnum, ToEnum,
                                              ASTNodeImporter::IDK_Basic);
    } else {
      CompleteDecl(ToEnum);
    }    
  } else if (ObjCInterfaceDecl *ToClass = dyn_cast<ObjCInterfaceDecl>(ToDC)) {
    ObjCInterfaceDecl *FromClass = cast<ObjCInterfaceDecl>(FromDC);
    if (ToClass->getDefinition()) {
      // Do nothing.
    } else if (ObjCInterfaceDecl *FromDef = FromClass->getDefinition()) {
      ASTNodeImporter(*this).ImportDefinition(FromDef, ToClass,
                                              ASTNodeImporter::IDK_Basic);
    } else {
      CompleteDecl(ToClass);
    }
  } else if (ObjCProtocolDecl *ToProto = dyn_cast<ObjCProtocolDecl>(ToDC)) {
    ObjCProtocolDecl *FromProto = cast<ObjCProtocolDecl>(FromDC);
    if (ToProto->getDefinition()) {
      // Do nothing.
    } else if (ObjCProtocolDecl *FromDef = FromProto->getDefinition()) {
      ASTNodeImporter(*this).ImportDefinition(FromDef, ToProto,
                                              ASTNodeImporter::IDK_Basic);
    } else {
      CompleteDecl(ToProto);
    }    
  }
  
  return ToDC;
}

Expr *ASTImporter::Import(Expr *FromE) {
  if (!FromE)
    return nullptr;

  return cast_or_null<Expr>(Import(cast<Stmt>(FromE)));
}

Stmt *ASTImporter::Import(Stmt *FromS) {
  if (!FromS)
    return nullptr;

  // Check whether we've already imported this declaration.  
  llvm::DenseMap<Stmt *, Stmt *>::iterator Pos = ImportedStmts.find(FromS);
  if (Pos != ImportedStmts.end())
    return Pos->second;
  
  // Import the type
  ASTNodeImporter Importer(*this);
  Stmt *ToS = Importer.Visit(FromS);
  if (!ToS)
    return nullptr;

  // Record the imported declaration.
  ImportedStmts[FromS] = ToS;
  return ToS;
}

NestedNameSpecifier *ASTImporter::Import(NestedNameSpecifier *FromNNS) {
  if (!FromNNS)
    return nullptr;

  NestedNameSpecifier *prefix = Import(FromNNS->getPrefix());

  switch (FromNNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    if (IdentifierInfo *II = Import(FromNNS->getAsIdentifier())) {
      return NestedNameSpecifier::Create(ToContext, prefix, II);
    }
    return nullptr;

  case NestedNameSpecifier::Namespace:
    if (NamespaceDecl *NS = 
          cast_or_null<NamespaceDecl>(Import(FromNNS->getAsNamespace()))) {
      return NestedNameSpecifier::Create(ToContext, prefix, NS);
    }
    return nullptr;

  case NestedNameSpecifier::NamespaceAlias:
    if (NamespaceAliasDecl *NSAD = 
          cast_or_null<NamespaceAliasDecl>(Import(FromNNS->getAsNamespaceAlias()))) {
      return NestedNameSpecifier::Create(ToContext, prefix, NSAD);
    }
    return nullptr;

  case NestedNameSpecifier::Global:
    return NestedNameSpecifier::GlobalSpecifier(ToContext);

  case NestedNameSpecifier::Super:
    if (CXXRecordDecl *RD =
            cast_or_null<CXXRecordDecl>(Import(FromNNS->getAsRecordDecl()))) {
      return NestedNameSpecifier::SuperSpecifier(ToContext, RD);
    }
    return nullptr;

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
      QualType T = Import(QualType(FromNNS->getAsType(), 0u));
      if (!T.isNull()) {
        bool bTemplate = FromNNS->getKind() == 
                         NestedNameSpecifier::TypeSpecWithTemplate;
        return NestedNameSpecifier::Create(ToContext, prefix, 
                                           bTemplate, T.getTypePtr());
      }
    }
      return nullptr;
  }

  llvm_unreachable("Invalid nested name specifier kind");
}

NestedNameSpecifierLoc ASTImporter::Import(NestedNameSpecifierLoc FromNNS) {
  // Copied from NestedNameSpecifier mostly.
  SmallVector<NestedNameSpecifierLoc , 8> NestedNames;
  NestedNameSpecifierLoc NNS = FromNNS;

  // Push each of the nested-name-specifiers's onto a stack for
  // serialization in reverse order.
  while (NNS) {
    NestedNames.push_back(NNS);
    NNS = NNS.getPrefix();
  }

  NestedNameSpecifierLocBuilder Builder;

  while (!NestedNames.empty()) {
    NNS = NestedNames.pop_back_val();
    NestedNameSpecifier *Spec = Import(NNS.getNestedNameSpecifier());
    if (!Spec)
      return NestedNameSpecifierLoc();

    NestedNameSpecifier::SpecifierKind Kind = Spec->getKind();
    switch (Kind) {
    case NestedNameSpecifier::Identifier:
      Builder.Extend(getToContext(),
                     Spec->getAsIdentifier(),
                     Import(NNS.getLocalBeginLoc()),
                     Import(NNS.getLocalEndLoc()));
      break;

    case NestedNameSpecifier::Namespace:
      Builder.Extend(getToContext(),
                     Spec->getAsNamespace(),
                     Import(NNS.getLocalBeginLoc()),
                     Import(NNS.getLocalEndLoc()));
      break;

    case NestedNameSpecifier::NamespaceAlias:
      Builder.Extend(getToContext(),
                     Spec->getAsNamespaceAlias(),
                     Import(NNS.getLocalBeginLoc()),
                     Import(NNS.getLocalEndLoc()));
      break;

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate: {
      TypeSourceInfo *TSI = getToContext().getTrivialTypeSourceInfo(
            QualType(Spec->getAsType(), 0));
      Builder.Extend(getToContext(),
                     Import(NNS.getLocalBeginLoc()),
                     TSI->getTypeLoc(),
                     Import(NNS.getLocalEndLoc()));
      break;
    }

    case NestedNameSpecifier::Global:
      Builder.MakeGlobal(getToContext(), Import(NNS.getLocalBeginLoc()));
      break;

    case NestedNameSpecifier::Super: {
      SourceRange ToRange = Import(NNS.getSourceRange());
      Builder.MakeSuper(getToContext(),
                        Spec->getAsRecordDecl(),
                        ToRange.getBegin(),
                        ToRange.getEnd());
    }
  }
  }

  return Builder.getWithLocInContext(getToContext());
}

TemplateName ASTImporter::Import(TemplateName From) {
  switch (From.getKind()) {
  case TemplateName::Template:
    if (TemplateDecl *ToTemplate
                = cast_or_null<TemplateDecl>(Import(From.getAsTemplateDecl())))
      return TemplateName(ToTemplate);
      
    return TemplateName();
      
  case TemplateName::OverloadedTemplate: {
    OverloadedTemplateStorage *FromStorage = From.getAsOverloadedTemplate();
    UnresolvedSet<2> ToTemplates;
    for (OverloadedTemplateStorage::iterator I = FromStorage->begin(),
                                             E = FromStorage->end();
         I != E; ++I) {
      if (NamedDecl *To = cast_or_null<NamedDecl>(Import(*I))) 
        ToTemplates.addDecl(To);
      else
        return TemplateName();
    }
    return ToContext.getOverloadedTemplateName(ToTemplates.begin(), 
                                               ToTemplates.end());
  }
      
  case TemplateName::QualifiedTemplate: {
    QualifiedTemplateName *QTN = From.getAsQualifiedTemplateName();
    NestedNameSpecifier *Qualifier = Import(QTN->getQualifier());
    if (!Qualifier)
      return TemplateName();
    
    if (TemplateDecl *ToTemplate
        = cast_or_null<TemplateDecl>(Import(From.getAsTemplateDecl())))
      return ToContext.getQualifiedTemplateName(Qualifier, 
                                                QTN->hasTemplateKeyword(), 
                                                ToTemplate);
    
    return TemplateName();
  }
  
  case TemplateName::DependentTemplate: {
    DependentTemplateName *DTN = From.getAsDependentTemplateName();
    NestedNameSpecifier *Qualifier = Import(DTN->getQualifier());
    if (!Qualifier)
      return TemplateName();
    
    if (DTN->isIdentifier()) {
      return ToContext.getDependentTemplateName(Qualifier, 
                                                Import(DTN->getIdentifier()));
    }
    
    return ToContext.getDependentTemplateName(Qualifier, DTN->getOperator());
  }

  case TemplateName::SubstTemplateTemplateParm: {
    SubstTemplateTemplateParmStorage *subst
      = From.getAsSubstTemplateTemplateParm();
    TemplateTemplateParmDecl *param
      = cast_or_null<TemplateTemplateParmDecl>(Import(subst->getParameter()));
    if (!param)
      return TemplateName();

    TemplateName replacement = Import(subst->getReplacement());
    if (replacement.isNull()) return TemplateName();
    
    return ToContext.getSubstTemplateTemplateParm(param, replacement);
  }
      
  case TemplateName::SubstTemplateTemplateParmPack: {
    SubstTemplateTemplateParmPackStorage *SubstPack
      = From.getAsSubstTemplateTemplateParmPack();
    TemplateTemplateParmDecl *Param
      = cast_or_null<TemplateTemplateParmDecl>(
                                        Import(SubstPack->getParameterPack()));
    if (!Param)
      return TemplateName();
    
    ASTNodeImporter Importer(*this);
    TemplateArgument ArgPack 
      = Importer.ImportTemplateArgument(SubstPack->getArgumentPack());
    if (ArgPack.isNull())
      return TemplateName();
    
    return ToContext.getSubstTemplateTemplateParmPack(Param, ArgPack);
  }
  }
  
  llvm_unreachable("Invalid template name kind");
}

SourceLocation ASTImporter::Import(SourceLocation FromLoc) {
  if (FromLoc.isInvalid())
    return SourceLocation();

  SourceManager &FromSM = FromContext.getSourceManager();
  
  // For now, map everything down to its file location, so that we
  // don't have to import macro expansions.
  // FIXME: Import macro expansions!
  FromLoc = FromSM.getFileLoc(FromLoc);
  std::pair<FileID, unsigned> Decomposed = FromSM.getDecomposedLoc(FromLoc);
  SourceManager &ToSM = ToContext.getSourceManager();
  FileID ToFileID = Import(Decomposed.first);
  if (ToFileID.isInvalid())
    return SourceLocation();
  SourceLocation ret = ToSM.getLocForStartOfFile(ToFileID)
                           .getLocWithOffset(Decomposed.second);
  return ret;
}

SourceRange ASTImporter::Import(SourceRange FromRange) {
  return SourceRange(Import(FromRange.getBegin()), Import(FromRange.getEnd()));
}

FileID ASTImporter::Import(FileID FromID) {
  llvm::DenseMap<FileID, FileID>::iterator Pos
    = ImportedFileIDs.find(FromID);
  if (Pos != ImportedFileIDs.end())
    return Pos->second;
  
  SourceManager &FromSM = FromContext.getSourceManager();
  SourceManager &ToSM = ToContext.getSourceManager();
  const SrcMgr::SLocEntry &FromSLoc = FromSM.getSLocEntry(FromID);
  assert(FromSLoc.isFile() && "Cannot handle macro expansions yet");
  
  // Include location of this file.
  SourceLocation ToIncludeLoc = Import(FromSLoc.getFile().getIncludeLoc());
  
  // Map the FileID for to the "to" source manager.
  FileID ToID;
  const SrcMgr::ContentCache *Cache = FromSLoc.getFile().getContentCache();
  if (Cache->OrigEntry && Cache->OrigEntry->getDir()) {
    // FIXME: We probably want to use getVirtualFile(), so we don't hit the
    // disk again
    // FIXME: We definitely want to re-use the existing MemoryBuffer, rather
    // than mmap the files several times.
    const FileEntry *Entry = ToFileManager.getFile(Cache->OrigEntry->getName());
    if (!Entry)
      return FileID();
    ToID = ToSM.createFileID(Entry, ToIncludeLoc, 
                             FromSLoc.getFile().getFileCharacteristic());
  } else {
    // FIXME: We want to re-use the existing MemoryBuffer!
    const llvm::MemoryBuffer *
        FromBuf = Cache->getBuffer(FromContext.getDiagnostics(), FromSM);
    std::unique_ptr<llvm::MemoryBuffer> ToBuf
      = llvm::MemoryBuffer::getMemBufferCopy(FromBuf->getBuffer(),
                                             FromBuf->getBufferIdentifier());
    ToID = ToSM.createFileID(std::move(ToBuf),
                             FromSLoc.getFile().getFileCharacteristic());
  }
  
  
  ImportedFileIDs[FromID] = ToID;
  return ToID;
}

CXXCtorInitializer *ASTImporter::Import(CXXCtorInitializer *From) {
  Expr *ToExpr = Import(From->getInit());
  if (!ToExpr && From->getInit())
    return nullptr;

  if (From->isBaseInitializer()) {
    TypeSourceInfo *ToTInfo = Import(From->getTypeSourceInfo());
    if (!ToTInfo && From->getTypeSourceInfo())
      return nullptr;

    return new (ToContext) CXXCtorInitializer(
        ToContext, ToTInfo, From->isBaseVirtual(), Import(From->getLParenLoc()),
        ToExpr, Import(From->getRParenLoc()),
        From->isPackExpansion() ? Import(From->getEllipsisLoc())
                                : SourceLocation());
  } else if (From->isMemberInitializer()) {
    FieldDecl *ToField =
        llvm::cast_or_null<FieldDecl>(Import(From->getMember()));
    if (!ToField && From->getMember())
      return nullptr;

    return new (ToContext) CXXCtorInitializer(
        ToContext, ToField, Import(From->getMemberLocation()),
        Import(From->getLParenLoc()), ToExpr, Import(From->getRParenLoc()));
  } else if (From->isIndirectMemberInitializer()) {
    IndirectFieldDecl *ToIField = llvm::cast_or_null<IndirectFieldDecl>(
        Import(From->getIndirectMember()));
    if (!ToIField && From->getIndirectMember())
      return nullptr;

    return new (ToContext) CXXCtorInitializer(
        ToContext, ToIField, Import(From->getMemberLocation()),
        Import(From->getLParenLoc()), ToExpr, Import(From->getRParenLoc()));
  } else if (From->isDelegatingInitializer()) {
    TypeSourceInfo *ToTInfo = Import(From->getTypeSourceInfo());
    if (!ToTInfo && From->getTypeSourceInfo())
      return nullptr;

    return new (ToContext)
        CXXCtorInitializer(ToContext, ToTInfo, Import(From->getLParenLoc()),
                           ToExpr, Import(From->getRParenLoc()));
  } else {
    return nullptr;
  }
}


CXXBaseSpecifier *ASTImporter::Import(const CXXBaseSpecifier *BaseSpec) {
  auto Pos = ImportedCXXBaseSpecifiers.find(BaseSpec);
  if (Pos != ImportedCXXBaseSpecifiers.end())
    return Pos->second;

  CXXBaseSpecifier *Imported = new (ToContext) CXXBaseSpecifier(
        Import(BaseSpec->getSourceRange()),
        BaseSpec->isVirtual(), BaseSpec->isBaseOfClass(),
        BaseSpec->getAccessSpecifierAsWritten(),
        Import(BaseSpec->getTypeSourceInfo()),
        Import(BaseSpec->getEllipsisLoc()));
  ImportedCXXBaseSpecifiers[BaseSpec] = Imported;
  return Imported;
}

void ASTImporter::ImportDefinition(Decl *From) {
  Decl *To = Import(From);
  if (!To)
    return;
  
  if (DeclContext *FromDC = cast<DeclContext>(From)) {
    ASTNodeImporter Importer(*this);
      
    if (RecordDecl *ToRecord = dyn_cast<RecordDecl>(To)) {
      if (!ToRecord->getDefinition()) {
        Importer.ImportDefinition(cast<RecordDecl>(FromDC), ToRecord, 
                                  ASTNodeImporter::IDK_Everything);
        return;
      }      
    }

    if (EnumDecl *ToEnum = dyn_cast<EnumDecl>(To)) {
      if (!ToEnum->getDefinition()) {
        Importer.ImportDefinition(cast<EnumDecl>(FromDC), ToEnum, 
                                  ASTNodeImporter::IDK_Everything);
        return;
      }      
    }
    
    if (ObjCInterfaceDecl *ToIFace = dyn_cast<ObjCInterfaceDecl>(To)) {
      if (!ToIFace->getDefinition()) {
        Importer.ImportDefinition(cast<ObjCInterfaceDecl>(FromDC), ToIFace,
                                  ASTNodeImporter::IDK_Everything);
        return;
      }
    }

    if (ObjCProtocolDecl *ToProto = dyn_cast<ObjCProtocolDecl>(To)) {
      if (!ToProto->getDefinition()) {
        Importer.ImportDefinition(cast<ObjCProtocolDecl>(FromDC), ToProto,
                                  ASTNodeImporter::IDK_Everything);
        return;
      }
    }
    
    Importer.ImportDeclContext(FromDC, true);
  }
}

DeclarationName ASTImporter::Import(DeclarationName FromName) {
  if (!FromName)
    return DeclarationName();

  switch (FromName.getNameKind()) {
  case DeclarationName::Identifier:
    return Import(FromName.getAsIdentifierInfo());

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    return Import(FromName.getObjCSelector());

  case DeclarationName::CXXConstructorName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXConstructorName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXDestructorName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXDestructorName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXDeductionGuideName: {
    TemplateDecl *Template = cast_or_null<TemplateDecl>(
        Import(FromName.getCXXDeductionGuideTemplate()));
    if (!Template)
      return DeclarationName();
    return ToContext.DeclarationNames.getCXXDeductionGuideName(Template);
  }

  case DeclarationName::CXXConversionFunctionName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXConversionFunctionName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXOperatorName:
    return ToContext.DeclarationNames.getCXXOperatorName(
                                          FromName.getCXXOverloadedOperator());

  case DeclarationName::CXXLiteralOperatorName:
    return ToContext.DeclarationNames.getCXXLiteralOperatorName(
                                   Import(FromName.getCXXLiteralIdentifier()));

  case DeclarationName::CXXUsingDirective:
    // FIXME: STATICS!
    return DeclarationName::getUsingDirectiveName();
  }

  llvm_unreachable("Invalid DeclarationName Kind!");
}

IdentifierInfo *ASTImporter::Import(const IdentifierInfo *FromId) {
  if (!FromId)
    return nullptr;

  IdentifierInfo *ToId = &ToContext.Idents.get(FromId->getName());

  if (!ToId->getBuiltinID() && FromId->getBuiltinID())
    ToId->setBuiltinID(FromId->getBuiltinID());

  return ToId;
}

Selector ASTImporter::Import(Selector FromSel) {
  if (FromSel.isNull())
    return Selector();

  SmallVector<IdentifierInfo *, 4> Idents;
  Idents.push_back(Import(FromSel.getIdentifierInfoForSlot(0)));
  for (unsigned I = 1, N = FromSel.getNumArgs(); I < N; ++I)
    Idents.push_back(Import(FromSel.getIdentifierInfoForSlot(I)));
  return ToContext.Selectors.getSelector(FromSel.getNumArgs(), Idents.data());
}

DeclarationName ASTImporter::HandleNameConflict(DeclarationName Name,
                                                DeclContext *DC,
                                                unsigned IDNS,
                                                NamedDecl **Decls,
                                                unsigned NumDecls) {
  return Name;
}

DiagnosticBuilder ASTImporter::ToDiag(SourceLocation Loc, unsigned DiagID) {
  if (LastDiagFromFrom)
    ToContext.getDiagnostics().notePriorDiagnosticFrom(
      FromContext.getDiagnostics());
  LastDiagFromFrom = false;
  return ToContext.getDiagnostics().Report(Loc, DiagID);
}

DiagnosticBuilder ASTImporter::FromDiag(SourceLocation Loc, unsigned DiagID) {
  if (!LastDiagFromFrom)
    FromContext.getDiagnostics().notePriorDiagnosticFrom(
      ToContext.getDiagnostics());
  LastDiagFromFrom = true;
  return FromContext.getDiagnostics().Report(Loc, DiagID);
}

void ASTImporter::CompleteDecl (Decl *D) {
  if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    if (!ID->getDefinition())
      ID->startDefinition();
  }
  else if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D)) {
    if (!PD->getDefinition())
      PD->startDefinition();
  }
  else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    if (!TD->getDefinition() && !TD->isBeingDefined()) {
      TD->startDefinition();
      TD->setCompleteDefinition(true);
    }
  }
  else {
    assert (0 && "CompleteDecl called on a Decl that can't be completed");
  }
}

Decl *ASTImporter::Imported(Decl *From, Decl *To) {
  if (From->hasAttrs()) {
    for (Attr *FromAttr : From->getAttrs())
      To->addAttr(FromAttr->clone(To->getASTContext()));
  }
  if (From->isUsed()) {
    To->setIsUsed();
  }
  if (From->isImplicit()) {
    To->setImplicit();
  }
  ImportedDecls[From] = To;
  return To;
}

bool ASTImporter::IsStructurallyEquivalent(QualType From, QualType To,
                                           bool Complain) {
  llvm::DenseMap<const Type *, const Type *>::iterator Pos
   = ImportedTypes.find(From.getTypePtr());
  if (Pos != ImportedTypes.end() && ToContext.hasSameType(Import(From), To))
    return true;

  StructuralEquivalenceContext Ctx(FromContext, ToContext, NonEquivalentDecls,
                                   false, Complain);
  return Ctx.IsStructurallyEquivalent(From, To);
}
