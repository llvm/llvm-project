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
    QualType VisitBuiltinType(const BuiltinType *T);
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
    // FIXME: DependentDecltypeType
    QualType VisitRecordType(const RecordType *T);
    QualType VisitEnumType(const EnumType *T);
    // FIXME: TemplateTypeParmType
    // FIXME: SubstTemplateTypeParmType
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
                         SourceLocation &Loc);
    void ImportDefinitionIfNeeded(Decl *FromD, Decl *ToD = 0);
    void ImportDeclarationNameLoc(const DeclarationNameInfo &From,
                                  DeclarationNameInfo& To);
    void ImportDeclContext(DeclContext *FromDC, bool ForceImport = false);
                        
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
    bool ImportDefinition(EnumDecl *From, EnumDecl *To,
                          ImportDefinitionKind Kind = IDK_Default);
    bool ImportDefinition(ObjCInterfaceDecl *From, ObjCInterfaceDecl *To,
                          ImportDefinitionKind Kind = IDK_Default);
    bool ImportDefinition(ObjCProtocolDecl *From, ObjCProtocolDecl *To,
                          ImportDefinitionKind Kind = IDK_Default);
    TemplateParameterList *ImportTemplateParameterList(
                                                 TemplateParameterList *Params);
    TemplateArgument ImportTemplateArgument(const TemplateArgument &From);
    bool ImportTemplateArguments(const TemplateArgument *FromArgs,
                                 unsigned NumFromArgs,
                               SmallVectorImpl<TemplateArgument> &ToArgs);
    bool IsStructuralMatch(RecordDecl *FromRecord, RecordDecl *ToRecord,
                           bool Complain = true);
    bool IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToRecord);
    bool IsStructuralMatch(ClassTemplateDecl *From, ClassTemplateDecl *To);
    Decl *VisitDecl(Decl *D);
    Decl *VisitTranslationUnitDecl(TranslationUnitDecl *D);
    Decl *VisitNamespaceDecl(NamespaceDecl *D);
    Decl *VisitTypedefNameDecl(TypedefNameDecl *D, bool IsAlias);
    Decl *VisitTypedefDecl(TypedefDecl *D);
    Decl *VisitTypeAliasDecl(TypeAliasDecl *D);
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
    Decl *VisitObjCIvarDecl(ObjCIvarDecl *D);
    Decl *VisitVarDecl(VarDecl *D);
    Decl *VisitImplicitParamDecl(ImplicitParamDecl *D);
    Decl *VisitParmVarDecl(ParmVarDecl *D);
    Decl *VisitObjCMethodDecl(ObjCMethodDecl *D);
    Decl *VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    Decl *VisitObjCProtocolDecl(ObjCProtocolDecl *D);
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
                            
    // Importing statements
    Stmt *VisitStmt(Stmt *S);

    // Importing expressions
    Expr *VisitExpr(Expr *E);
    Expr *VisitDeclRefExpr(DeclRefExpr *E);
    Expr *VisitIntegerLiteral(IntegerLiteral *E);
    Expr *VisitCharacterLiteral(CharacterLiteral *E);
    Expr *VisitParenExpr(ParenExpr *E);
    Expr *VisitUnaryOperator(UnaryOperator *E);
    Expr *VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
    Expr *VisitBinaryOperator(BinaryOperator *E);
    Expr *VisitCompoundAssignOperator(CompoundAssignOperator *E);
    Expr *VisitImplicitCastExpr(ImplicitCastExpr *E);
    Expr *VisitCStyleCastExpr(CStyleCastExpr *E);
  };
}
using namespace clang;

//----------------------------------------------------------------------------
// Structural Equivalence
//----------------------------------------------------------------------------

namespace {
  struct StructuralEquivalenceContext {
    /// \brief AST contexts for which we are checking structural equivalence.
    ASTContext &C1, &C2;
    
    /// \brief The set of "tentative" equivalences between two canonical 
    /// declarations, mapping from a declaration in the first context to the
    /// declaration in the second context that we believe to be equivalent.
    llvm::DenseMap<Decl *, Decl *> TentativeEquivalences;
    
    /// \brief Queue of declarations in the first context whose equivalence
    /// with a declaration in the second context still needs to be verified.
    std::deque<Decl *> DeclsToCheck;
    
    /// \brief Declaration (from, to) pairs that are known not to be equivalent
    /// (which we have already complained about).
    llvm::DenseSet<std::pair<Decl *, Decl *> > &NonEquivalentDecls;
    
    /// \brief Whether we're being strict about the spelling of types when 
    /// unifying two types.
    bool StrictTypeSpelling;

    /// \brief Whether to complain about failures.
    bool Complain;

    StructuralEquivalenceContext(ASTContext &C1, ASTContext &C2,
               llvm::DenseSet<std::pair<Decl *, Decl *> > &NonEquivalentDecls,
                                 bool StrictTypeSpelling = false,
                                 bool Complain = true)
      : C1(C1), C2(C2), NonEquivalentDecls(NonEquivalentDecls),
        StrictTypeSpelling(StrictTypeSpelling), Complain(Complain) { }

    /// \brief Determine whether the two declarations are structurally
    /// equivalent.
    bool IsStructurallyEquivalent(Decl *D1, Decl *D2);
    
    /// \brief Determine whether the two types are structurally equivalent.
    bool IsStructurallyEquivalent(QualType T1, QualType T2);

  private:
    /// \brief Finish checking all of the structural equivalences.
    ///
    /// \returns true if an error occurred, false otherwise.
    bool Finish();
    
  public:
    DiagnosticBuilder Diag1(SourceLocation Loc, unsigned DiagID) {
      assert(Complain && "Not allowed to complain");
      return C1.getDiagnostics().Report(Loc, DiagID);
    }

    DiagnosticBuilder Diag2(SourceLocation Loc, unsigned DiagID) {
      assert(Complain && "Not allowed to complain");
      return C2.getDiagnostics().Report(Loc, DiagID);
    }
  };
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     QualType T1, QualType T2);
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     Decl *D1, Decl *D2);

/// \brief Determine structural equivalence of two expressions.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     Expr *E1, Expr *E2) {
  if (!E1 || !E2)
    return E1 == E2;
  
  // FIXME: Actually perform a structural comparison!
  return true;
}

/// \brief Determine whether two identifiers are equivalent.
static bool IsStructurallyEquivalent(const IdentifierInfo *Name1,
                                     const IdentifierInfo *Name2) {
  if (!Name1 || !Name2)
    return Name1 == Name2;
  
  return Name1->getName() == Name2->getName();
}

/// \brief Determine whether two nested-name-specifiers are equivalent.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     NestedNameSpecifier *NNS1,
                                     NestedNameSpecifier *NNS2) {
  // FIXME: Implement!
  return true;
}

/// \brief Determine whether two template arguments are equivalent.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     const TemplateArgument &Arg1,
                                     const TemplateArgument &Arg2) {
  if (Arg1.getKind() != Arg2.getKind())
    return false;

  switch (Arg1.getKind()) {
  case TemplateArgument::Null:
    return true;
      
  case TemplateArgument::Type:
    return Context.IsStructurallyEquivalent(Arg1.getAsType(), Arg2.getAsType());

  case TemplateArgument::Integral:
    if (!Context.IsStructurallyEquivalent(Arg1.getIntegralType(), 
                                          Arg2.getIntegralType()))
      return false;
    
    return llvm::APSInt::isSameValue(Arg1.getAsIntegral(), Arg2.getAsIntegral());
      
  case TemplateArgument::Declaration:
    return Context.IsStructurallyEquivalent(Arg1.getAsDecl(), Arg2.getAsDecl());

  case TemplateArgument::NullPtr:
    return true; // FIXME: Is this correct?

  case TemplateArgument::Template:
    return IsStructurallyEquivalent(Context, 
                                    Arg1.getAsTemplate(), 
                                    Arg2.getAsTemplate());

  case TemplateArgument::TemplateExpansion:
    return IsStructurallyEquivalent(Context, 
                                    Arg1.getAsTemplateOrTemplatePattern(), 
                                    Arg2.getAsTemplateOrTemplatePattern());

  case TemplateArgument::Expression:
    return IsStructurallyEquivalent(Context, 
                                    Arg1.getAsExpr(), Arg2.getAsExpr());
      
  case TemplateArgument::Pack:
    if (Arg1.pack_size() != Arg2.pack_size())
      return false;
      
    for (unsigned I = 0, N = Arg1.pack_size(); I != N; ++I)
      if (!IsStructurallyEquivalent(Context, 
                                    Arg1.pack_begin()[I],
                                    Arg2.pack_begin()[I]))
        return false;
      
    return true;
  }
  
  llvm_unreachable("Invalid template argument kind");
}

/// \brief Determine structural equivalence for the common part of array 
/// types.
static bool IsArrayStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                          const ArrayType *Array1, 
                                          const ArrayType *Array2) {
  if (!IsStructurallyEquivalent(Context, 
                                Array1->getElementType(), 
                                Array2->getElementType()))
    return false;
  if (Array1->getSizeModifier() != Array2->getSizeModifier())
    return false;
  if (Array1->getIndexTypeQualifiers() != Array2->getIndexTypeQualifiers())
    return false;
  
  return true;
}

/// \brief Determine structural equivalence of two types.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     QualType T1, QualType T2) {
  if (T1.isNull() || T2.isNull())
    return T1.isNull() && T2.isNull();
  
  if (!Context.StrictTypeSpelling) {
    // We aren't being strict about token-to-token equivalence of types,
    // so map down to the canonical type.
    T1 = Context.C1.getCanonicalType(T1);
    T2 = Context.C2.getCanonicalType(T2);
  }
  
  if (T1.getQualifiers() != T2.getQualifiers())
    return false;
  
  Type::TypeClass TC = T1->getTypeClass();
  
  if (T1->getTypeClass() != T2->getTypeClass()) {
    // Compare function types with prototypes vs. without prototypes as if
    // both did not have prototypes.
    if (T1->getTypeClass() == Type::FunctionProto &&
        T2->getTypeClass() == Type::FunctionNoProto)
      TC = Type::FunctionNoProto;
    else if (T1->getTypeClass() == Type::FunctionNoProto &&
             T2->getTypeClass() == Type::FunctionProto)
      TC = Type::FunctionNoProto;
    else
      return false;
  }
  
  switch (TC) {
  case Type::Builtin:
    // FIXME: Deal with Char_S/Char_U. 
    if (cast<BuiltinType>(T1)->getKind() != cast<BuiltinType>(T2)->getKind())
      return false;
    break;
  
  case Type::Complex:
    if (!IsStructurallyEquivalent(Context,
                                  cast<ComplexType>(T1)->getElementType(),
                                  cast<ComplexType>(T2)->getElementType()))
      return false;
    break;
  
  case Type::Pointer:
    if (!IsStructurallyEquivalent(Context,
                                  cast<PointerType>(T1)->getPointeeType(),
                                  cast<PointerType>(T2)->getPointeeType()))
      return false;
    break;

  case Type::BlockPointer:
    if (!IsStructurallyEquivalent(Context,
                                  cast<BlockPointerType>(T1)->getPointeeType(),
                                  cast<BlockPointerType>(T2)->getPointeeType()))
      return false;
    break;

  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType *Ref1 = cast<ReferenceType>(T1);
    const ReferenceType *Ref2 = cast<ReferenceType>(T2);
    if (Ref1->isSpelledAsLValue() != Ref2->isSpelledAsLValue())
      return false;
    if (Ref1->isInnerRef() != Ref2->isInnerRef())
      return false;
    if (!IsStructurallyEquivalent(Context,
                                  Ref1->getPointeeTypeAsWritten(),
                                  Ref2->getPointeeTypeAsWritten()))
      return false;
    break;
  }
      
  case Type::MemberPointer: {
    const MemberPointerType *MemPtr1 = cast<MemberPointerType>(T1);
    const MemberPointerType *MemPtr2 = cast<MemberPointerType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  MemPtr1->getPointeeType(),
                                  MemPtr2->getPointeeType()))
      return false;
    if (!IsStructurallyEquivalent(Context,
                                  QualType(MemPtr1->getClass(), 0),
                                  QualType(MemPtr2->getClass(), 0)))
      return false;
    break;
  }
      
  case Type::ConstantArray: {
    const ConstantArrayType *Array1 = cast<ConstantArrayType>(T1);
    const ConstantArrayType *Array2 = cast<ConstantArrayType>(T2);
    if (!llvm::APInt::isSameValue(Array1->getSize(), Array2->getSize()))
      return false;
    
    if (!IsArrayStructurallyEquivalent(Context, Array1, Array2))
      return false;
    break;
  }

  case Type::IncompleteArray:
    if (!IsArrayStructurallyEquivalent(Context, 
                                       cast<ArrayType>(T1), 
                                       cast<ArrayType>(T2)))
      return false;
    break;
      
  case Type::VariableArray: {
    const VariableArrayType *Array1 = cast<VariableArrayType>(T1);
    const VariableArrayType *Array2 = cast<VariableArrayType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Array1->getSizeExpr(), Array2->getSizeExpr()))
      return false;
    
    if (!IsArrayStructurallyEquivalent(Context, Array1, Array2))
      return false;
    
    break;
  }
  
  case Type::DependentSizedArray: {
    const DependentSizedArrayType *Array1 = cast<DependentSizedArrayType>(T1);
    const DependentSizedArrayType *Array2 = cast<DependentSizedArrayType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Array1->getSizeExpr(), Array2->getSizeExpr()))
      return false;
    
    if (!IsArrayStructurallyEquivalent(Context, Array1, Array2))
      return false;
    
    break;
  }
      
  case Type::DependentSizedExtVector: {
    const DependentSizedExtVectorType *Vec1
      = cast<DependentSizedExtVectorType>(T1);
    const DependentSizedExtVectorType *Vec2
      = cast<DependentSizedExtVectorType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Vec1->getSizeExpr(), Vec2->getSizeExpr()))
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Vec1->getElementType(), 
                                  Vec2->getElementType()))
      return false;
    break;
  }
   
  case Type::Vector: 
  case Type::ExtVector: {
    const VectorType *Vec1 = cast<VectorType>(T1);
    const VectorType *Vec2 = cast<VectorType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Vec1->getElementType(),
                                  Vec2->getElementType()))
      return false;
    if (Vec1->getNumElements() != Vec2->getNumElements())
      return false;
    if (Vec1->getVectorKind() != Vec2->getVectorKind())
      return false;
    break;
  }

  case Type::FunctionProto: {
    const FunctionProtoType *Proto1 = cast<FunctionProtoType>(T1);
    const FunctionProtoType *Proto2 = cast<FunctionProtoType>(T2);
    if (Proto1->getNumArgs() != Proto2->getNumArgs())
      return false;
    for (unsigned I = 0, N = Proto1->getNumArgs(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context, 
                                    Proto1->getArgType(I),
                                    Proto2->getArgType(I)))
        return false;
    }
    if (Proto1->isVariadic() != Proto2->isVariadic())
      return false;
    if (Proto1->getExceptionSpecType() != Proto2->getExceptionSpecType())
      return false;
    if (Proto1->getExceptionSpecType() == EST_Dynamic) {
      if (Proto1->getNumExceptions() != Proto2->getNumExceptions())
        return false;
      for (unsigned I = 0, N = Proto1->getNumExceptions(); I != N; ++I) {
        if (!IsStructurallyEquivalent(Context,
                                      Proto1->getExceptionType(I),
                                      Proto2->getExceptionType(I)))
          return false;
      }
    } else if (Proto1->getExceptionSpecType() == EST_ComputedNoexcept) {
      if (!IsStructurallyEquivalent(Context,
                                    Proto1->getNoexceptExpr(),
                                    Proto2->getNoexceptExpr()))
        return false;
    }
    if (Proto1->getTypeQuals() != Proto2->getTypeQuals())
      return false;
    
    // Fall through to check the bits common with FunctionNoProtoType.
  }
      
  case Type::FunctionNoProto: {
    const FunctionType *Function1 = cast<FunctionType>(T1);
    const FunctionType *Function2 = cast<FunctionType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Function1->getResultType(),
                                  Function2->getResultType()))
      return false;
      if (Function1->getExtInfo() != Function2->getExtInfo())
        return false;
    break;
  }
   
  case Type::UnresolvedUsing:
    if (!IsStructurallyEquivalent(Context,
                                  cast<UnresolvedUsingType>(T1)->getDecl(),
                                  cast<UnresolvedUsingType>(T2)->getDecl()))
      return false;
      
    break;

  case Type::Attributed:
    if (!IsStructurallyEquivalent(Context,
                                  cast<AttributedType>(T1)->getModifiedType(),
                                  cast<AttributedType>(T2)->getModifiedType()))
      return false;
    if (!IsStructurallyEquivalent(Context,
                                cast<AttributedType>(T1)->getEquivalentType(),
                                cast<AttributedType>(T2)->getEquivalentType()))
      return false;
    break;
      
  case Type::Paren:
    if (!IsStructurallyEquivalent(Context,
                                  cast<ParenType>(T1)->getInnerType(),
                                  cast<ParenType>(T2)->getInnerType()))
      return false;
    break;

  case Type::Typedef:
    if (!IsStructurallyEquivalent(Context,
                                  cast<TypedefType>(T1)->getDecl(),
                                  cast<TypedefType>(T2)->getDecl()))
      return false;
    break;
      
  case Type::TypeOfExpr:
    if (!IsStructurallyEquivalent(Context,
                                cast<TypeOfExprType>(T1)->getUnderlyingExpr(),
                                cast<TypeOfExprType>(T2)->getUnderlyingExpr()))
      return false;
    break;
      
  case Type::TypeOf:
    if (!IsStructurallyEquivalent(Context,
                                  cast<TypeOfType>(T1)->getUnderlyingType(),
                                  cast<TypeOfType>(T2)->getUnderlyingType()))
      return false;
    break;

  case Type::UnaryTransform:
    if (!IsStructurallyEquivalent(Context,
                             cast<UnaryTransformType>(T1)->getUnderlyingType(),
                             cast<UnaryTransformType>(T1)->getUnderlyingType()))
      return false;
    break;

  case Type::Decltype:
    if (!IsStructurallyEquivalent(Context,
                                  cast<DecltypeType>(T1)->getUnderlyingExpr(),
                                  cast<DecltypeType>(T2)->getUnderlyingExpr()))
      return false;
    break;

  case Type::Auto:
    if (!IsStructurallyEquivalent(Context,
                                  cast<AutoType>(T1)->getDeducedType(),
                                  cast<AutoType>(T2)->getDeducedType()))
      return false;
    break;

  case Type::Record:
  case Type::Enum:
    if (!IsStructurallyEquivalent(Context,
                                  cast<TagType>(T1)->getDecl(),
                                  cast<TagType>(T2)->getDecl()))
      return false;
    break;

  case Type::TemplateTypeParm: {
    const TemplateTypeParmType *Parm1 = cast<TemplateTypeParmType>(T1);
    const TemplateTypeParmType *Parm2 = cast<TemplateTypeParmType>(T2);
    if (Parm1->getDepth() != Parm2->getDepth())
      return false;
    if (Parm1->getIndex() != Parm2->getIndex())
      return false;
    if (Parm1->isParameterPack() != Parm2->isParameterPack())
      return false;
    
    // Names of template type parameters are never significant.
    break;
  }
      
  case Type::SubstTemplateTypeParm: {
    const SubstTemplateTypeParmType *Subst1
      = cast<SubstTemplateTypeParmType>(T1);
    const SubstTemplateTypeParmType *Subst2
      = cast<SubstTemplateTypeParmType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  QualType(Subst1->getReplacedParameter(), 0),
                                  QualType(Subst2->getReplacedParameter(), 0)))
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Subst1->getReplacementType(),
                                  Subst2->getReplacementType()))
      return false;
    break;
  }

  case Type::SubstTemplateTypeParmPack: {
    const SubstTemplateTypeParmPackType *Subst1
      = cast<SubstTemplateTypeParmPackType>(T1);
    const SubstTemplateTypeParmPackType *Subst2
      = cast<SubstTemplateTypeParmPackType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  QualType(Subst1->getReplacedParameter(), 0),
                                  QualType(Subst2->getReplacedParameter(), 0)))
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Subst1->getArgumentPack(),
                                  Subst2->getArgumentPack()))
      return false;
    break;
  }
  case Type::TemplateSpecialization: {
    const TemplateSpecializationType *Spec1
      = cast<TemplateSpecializationType>(T1);
    const TemplateSpecializationType *Spec2
      = cast<TemplateSpecializationType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  Spec1->getTemplateName(),
                                  Spec2->getTemplateName()))
      return false;
    if (Spec1->getNumArgs() != Spec2->getNumArgs())
      return false;
    for (unsigned I = 0, N = Spec1->getNumArgs(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context, 
                                    Spec1->getArg(I), Spec2->getArg(I)))
        return false;
    }
    break;
  }
      
  case Type::Elaborated: {
    const ElaboratedType *Elab1 = cast<ElaboratedType>(T1);
    const ElaboratedType *Elab2 = cast<ElaboratedType>(T2);
    // CHECKME: what if a keyword is ETK_None or ETK_typename ?
    if (Elab1->getKeyword() != Elab2->getKeyword())
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Elab1->getQualifier(), 
                                  Elab2->getQualifier()))
      return false;
    if (!IsStructurallyEquivalent(Context,
                                  Elab1->getNamedType(),
                                  Elab2->getNamedType()))
      return false;
    break;
  }

  case Type::InjectedClassName: {
    const InjectedClassNameType *Inj1 = cast<InjectedClassNameType>(T1);
    const InjectedClassNameType *Inj2 = cast<InjectedClassNameType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  Inj1->getInjectedSpecializationType(),
                                  Inj2->getInjectedSpecializationType()))
      return false;
    break;
  }

  case Type::DependentName: {
    const DependentNameType *Typename1 = cast<DependentNameType>(T1);
    const DependentNameType *Typename2 = cast<DependentNameType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Typename1->getQualifier(),
                                  Typename2->getQualifier()))
      return false;
    if (!IsStructurallyEquivalent(Typename1->getIdentifier(),
                                  Typename2->getIdentifier()))
      return false;
    
    break;
  }
  
  case Type::DependentTemplateSpecialization: {
    const DependentTemplateSpecializationType *Spec1 =
      cast<DependentTemplateSpecializationType>(T1);
    const DependentTemplateSpecializationType *Spec2 =
      cast<DependentTemplateSpecializationType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Spec1->getQualifier(),
                                  Spec2->getQualifier()))
      return false;
    if (!IsStructurallyEquivalent(Spec1->getIdentifier(),
                                  Spec2->getIdentifier()))
      return false;
    if (Spec1->getNumArgs() != Spec2->getNumArgs())
      return false;
    for (unsigned I = 0, N = Spec1->getNumArgs(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context,
                                    Spec1->getArg(I), Spec2->getArg(I)))
        return false;
    }
    break;
  }

  case Type::PackExpansion:
    if (!IsStructurallyEquivalent(Context,
                                  cast<PackExpansionType>(T1)->getPattern(),
                                  cast<PackExpansionType>(T2)->getPattern()))
      return false;
    break;

  case Type::ObjCInterface: {
    const ObjCInterfaceType *Iface1 = cast<ObjCInterfaceType>(T1);
    const ObjCInterfaceType *Iface2 = cast<ObjCInterfaceType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Iface1->getDecl(), Iface2->getDecl()))
      return false;
    break;
  }

  case Type::ObjCObject: {
    const ObjCObjectType *Obj1 = cast<ObjCObjectType>(T1);
    const ObjCObjectType *Obj2 = cast<ObjCObjectType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  Obj1->getBaseType(),
                                  Obj2->getBaseType()))
      return false;
    if (Obj1->getNumProtocols() != Obj2->getNumProtocols())
      return false;
    for (unsigned I = 0, N = Obj1->getNumProtocols(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context,
                                    Obj1->getProtocol(I),
                                    Obj2->getProtocol(I)))
        return false;
    }
    break;
  }

  case Type::ObjCObjectPointer: {
    const ObjCObjectPointerType *Ptr1 = cast<ObjCObjectPointerType>(T1);
    const ObjCObjectPointerType *Ptr2 = cast<ObjCObjectPointerType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Ptr1->getPointeeType(),
                                  Ptr2->getPointeeType()))
      return false;
    break;
  }

  case Type::Atomic: {
    if (!IsStructurallyEquivalent(Context,
                                  cast<AtomicType>(T1)->getValueType(),
                                  cast<AtomicType>(T2)->getValueType()))
      return false;
    break;
  }

  } // end switch

  return true;
}

/// \brief Determine structural equivalence of two fields.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     FieldDecl *Field1, FieldDecl *Field2) {
  RecordDecl *Owner2 = cast<RecordDecl>(Field2->getDeclContext());

  // For anonymous structs/unions, match up the anonymous struct/union type
  // declarations directly, so that we don't go off searching for anonymous
  // types
  if (Field1->isAnonymousStructOrUnion() &&
      Field2->isAnonymousStructOrUnion()) {
    RecordDecl *D1 = Field1->getType()->castAs<RecordType>()->getDecl();
    RecordDecl *D2 = Field2->getType()->castAs<RecordType>()->getDecl();
    return IsStructurallyEquivalent(Context, D1, D2);
  }

  if (!IsStructurallyEquivalent(Context,
                                Field1->getType(), Field2->getType())) {
    if (Context.Complain) {
      Context.Diag2(Owner2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(Owner2);
      Context.Diag2(Field2->getLocation(), diag::note_odr_field)
        << Field2->getDeclName() << Field2->getType();
      Context.Diag1(Field1->getLocation(), diag::note_odr_field)
        << Field1->getDeclName() << Field1->getType();
    }
    return false;
  }
  
  if (Field1->isBitField() != Field2->isBitField()) {
    if (Context.Complain) {
      Context.Diag2(Owner2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(Owner2);
      if (Field1->isBitField()) {
        Context.Diag1(Field1->getLocation(), diag::note_odr_bit_field)
        << Field1->getDeclName() << Field1->getType()
        << Field1->getBitWidthValue(Context.C1);
        Context.Diag2(Field2->getLocation(), diag::note_odr_not_bit_field)
        << Field2->getDeclName();
      } else {
        Context.Diag2(Field2->getLocation(), diag::note_odr_bit_field)
        << Field2->getDeclName() << Field2->getType()
        << Field2->getBitWidthValue(Context.C2);
        Context.Diag1(Field1->getLocation(), diag::note_odr_not_bit_field)
        << Field1->getDeclName();
      }
    }
    return false;
  }
  
  if (Field1->isBitField()) {
    // Make sure that the bit-fields are the same length.
    unsigned Bits1 = Field1->getBitWidthValue(Context.C1);
    unsigned Bits2 = Field2->getBitWidthValue(Context.C2);
    
    if (Bits1 != Bits2) {
      if (Context.Complain) {
        Context.Diag2(Owner2->getLocation(), diag::warn_odr_tag_type_inconsistent)
          << Context.C2.getTypeDeclType(Owner2);
        Context.Diag2(Field2->getLocation(), diag::note_odr_bit_field)
          << Field2->getDeclName() << Field2->getType() << Bits2;
        Context.Diag1(Field1->getLocation(), diag::note_odr_bit_field)
          << Field1->getDeclName() << Field1->getType() << Bits1;
      }
      return false;
    }
  }

  return true;
}

/// \brief Find the index of the given anonymous struct/union within its
/// context.
///
/// \returns Returns the index of this anonymous struct/union in its context,
/// including the next assigned index (if none of them match). Returns an
/// empty option if the context is not a record, i.e.. if the anonymous
/// struct/union is at namespace or block scope.
static llvm::Optional<unsigned>
findAnonymousStructOrUnionIndex(RecordDecl *Anon) {
  ASTContext &Context = Anon->getASTContext();
  QualType AnonTy = Context.getRecordType(Anon);

  RecordDecl *Owner = dyn_cast<RecordDecl>(Anon->getDeclContext());
  if (!Owner)
    return llvm::Optional<unsigned>();

  unsigned Index = 0;
  for (DeclContext::decl_iterator D = Owner->noload_decls_begin(),
                               DEnd = Owner->noload_decls_end();
       D != DEnd; ++D) {
    FieldDecl *F = dyn_cast<FieldDecl>(*D);
    if (!F || !F->isAnonymousStructOrUnion())
      continue;

    if (Context.hasSameType(F->getType(), AnonTy))
      break;

    ++Index;
  }

  return Index;
}

/// \brief Determine structural equivalence of two records.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     RecordDecl *D1, RecordDecl *D2) {
  if (D1->isUnion() != D2->isUnion()) {
    if (Context.Complain) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      Context.Diag1(D1->getLocation(), diag::note_odr_tag_kind_here)
        << D1->getDeclName() << (unsigned)D1->getTagKind();
    }
    return false;
  }

  if (D1->isAnonymousStructOrUnion() && D2->isAnonymousStructOrUnion()) {
    // If both anonymous structs/unions are in a record context, make sure
    // they occur in the same location in the context records.
    if (llvm::Optional<unsigned> Index1
          = findAnonymousStructOrUnionIndex(D1)) {
      if (llvm::Optional<unsigned> Index2
            = findAnonymousStructOrUnionIndex(D2)) {
        if (*Index1 != *Index2)
          return false;
      }
    }
  }

  // If both declarations are class template specializations, we know
  // the ODR applies, so check the template and template arguments.
  ClassTemplateSpecializationDecl *Spec1
    = dyn_cast<ClassTemplateSpecializationDecl>(D1);
  ClassTemplateSpecializationDecl *Spec2
    = dyn_cast<ClassTemplateSpecializationDecl>(D2);
  if (Spec1 && Spec2) {
    // Check that the specialized templates are the same.
    if (!IsStructurallyEquivalent(Context, Spec1->getSpecializedTemplate(),
                                  Spec2->getSpecializedTemplate()))
      return false;
    
    // Check that the template arguments are the same.
    if (Spec1->getTemplateArgs().size() != Spec2->getTemplateArgs().size())
      return false;
    
    for (unsigned I = 0, N = Spec1->getTemplateArgs().size(); I != N; ++I)
      if (!IsStructurallyEquivalent(Context, 
                                    Spec1->getTemplateArgs().get(I),
                                    Spec2->getTemplateArgs().get(I)))
        return false;
  }  
  // If one is a class template specialization and the other is not, these
  // structures are different.
  else if (Spec1 || Spec2)
    return false;

  // Compare the definitions of these two records. If either or both are
  // incomplete, we assume that they are equivalent.
  D1 = D1->getDefinition();
  D2 = D2->getDefinition();
  if (!D1 || !D2)
    return true;
  
  if (CXXRecordDecl *D1CXX = dyn_cast<CXXRecordDecl>(D1)) {
    if (CXXRecordDecl *D2CXX = dyn_cast<CXXRecordDecl>(D2)) {
      if (D1CXX->getNumBases() != D2CXX->getNumBases()) {
        if (Context.Complain) {
          Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
            << Context.C2.getTypeDeclType(D2);
          Context.Diag2(D2->getLocation(), diag::note_odr_number_of_bases)
            << D2CXX->getNumBases();
          Context.Diag1(D1->getLocation(), diag::note_odr_number_of_bases)
            << D1CXX->getNumBases();
        }
        return false;
      }
      
      // Check the base classes. 
      for (CXXRecordDecl::base_class_iterator Base1 = D1CXX->bases_begin(), 
                                           BaseEnd1 = D1CXX->bases_end(),
                                                Base2 = D2CXX->bases_begin();
           Base1 != BaseEnd1;
           ++Base1, ++Base2) {        
        if (!IsStructurallyEquivalent(Context, 
                                      Base1->getType(), Base2->getType())) {
          if (Context.Complain) {
            Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
              << Context.C2.getTypeDeclType(D2);
            Context.Diag2(Base2->getLocStart(), diag::note_odr_base)
              << Base2->getType()
              << Base2->getSourceRange();
            Context.Diag1(Base1->getLocStart(), diag::note_odr_base)
              << Base1->getType()
              << Base1->getSourceRange();
          }
          return false;
        }
        
        // Check virtual vs. non-virtual inheritance mismatch.
        if (Base1->isVirtual() != Base2->isVirtual()) {
          if (Context.Complain) {
            Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
              << Context.C2.getTypeDeclType(D2);
            Context.Diag2(Base2->getLocStart(),
                          diag::note_odr_virtual_base)
              << Base2->isVirtual() << Base2->getSourceRange();
            Context.Diag1(Base1->getLocStart(), diag::note_odr_base)
              << Base1->isVirtual()
              << Base1->getSourceRange();
          }
          return false;
        }
      }
    } else if (D1CXX->getNumBases() > 0) {
      if (Context.Complain) {
        Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
          << Context.C2.getTypeDeclType(D2);
        const CXXBaseSpecifier *Base1 = D1CXX->bases_begin();
        Context.Diag1(Base1->getLocStart(), diag::note_odr_base)
          << Base1->getType()
          << Base1->getSourceRange();
        Context.Diag2(D2->getLocation(), diag::note_odr_missing_base);
      }
      return false;
    }
  }
  
  // Check the fields for consistency.
  RecordDecl::field_iterator Field2 = D2->field_begin(),
                             Field2End = D2->field_end();
  for (RecordDecl::field_iterator Field1 = D1->field_begin(),
                                  Field1End = D1->field_end();
       Field1 != Field1End;
       ++Field1, ++Field2) {
    if (Field2 == Field2End) {
      if (Context.Complain) {
        Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
          << Context.C2.getTypeDeclType(D2);
        Context.Diag1(Field1->getLocation(), diag::note_odr_field)
          << Field1->getDeclName() << Field1->getType();
        Context.Diag2(D2->getLocation(), diag::note_odr_missing_field);
      }
      return false;
    }
    
    if (!IsStructurallyEquivalent(Context, *Field1, *Field2))
      return false;    
  }
  
  if (Field2 != Field2End) {
    if (Context.Complain) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      Context.Diag2(Field2->getLocation(), diag::note_odr_field)
        << Field2->getDeclName() << Field2->getType();
      Context.Diag1(D1->getLocation(), diag::note_odr_missing_field);
    }
    return false;
  }
  
  return true;
}
     
/// \brief Determine structural equivalence of two enums.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     EnumDecl *D1, EnumDecl *D2) {
  EnumDecl::enumerator_iterator EC2 = D2->enumerator_begin(),
                             EC2End = D2->enumerator_end();
  for (EnumDecl::enumerator_iterator EC1 = D1->enumerator_begin(),
                                  EC1End = D1->enumerator_end();
       EC1 != EC1End; ++EC1, ++EC2) {
    if (EC2 == EC2End) {
      if (Context.Complain) {
        Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
          << Context.C2.getTypeDeclType(D2);
        Context.Diag1(EC1->getLocation(), diag::note_odr_enumerator)
          << EC1->getDeclName() 
          << EC1->getInitVal().toString(10);
        Context.Diag2(D2->getLocation(), diag::note_odr_missing_enumerator);
      }
      return false;
    }
    
    llvm::APSInt Val1 = EC1->getInitVal();
    llvm::APSInt Val2 = EC2->getInitVal();
    if (!llvm::APSInt::isSameValue(Val1, Val2) || 
        !IsStructurallyEquivalent(EC1->getIdentifier(), EC2->getIdentifier())) {
      if (Context.Complain) {
        Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
          << Context.C2.getTypeDeclType(D2);
        Context.Diag2(EC2->getLocation(), diag::note_odr_enumerator)
          << EC2->getDeclName() 
          << EC2->getInitVal().toString(10);
        Context.Diag1(EC1->getLocation(), diag::note_odr_enumerator)
          << EC1->getDeclName() 
          << EC1->getInitVal().toString(10);
      }
      return false;
    }
  }
  
  if (EC2 != EC2End) {
    if (Context.Complain) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      Context.Diag2(EC2->getLocation(), diag::note_odr_enumerator)
        << EC2->getDeclName() 
        << EC2->getInitVal().toString(10);
      Context.Diag1(D1->getLocation(), diag::note_odr_missing_enumerator);
    }
    return false;
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     TemplateParameterList *Params1,
                                     TemplateParameterList *Params2) {
  if (Params1->size() != Params2->size()) {
    if (Context.Complain) {
      Context.Diag2(Params2->getTemplateLoc(), 
                    diag::err_odr_different_num_template_parameters)
        << Params1->size() << Params2->size();
      Context.Diag1(Params1->getTemplateLoc(), 
                    diag::note_odr_template_parameter_list);
    }
    return false;
  }
  
  for (unsigned I = 0, N = Params1->size(); I != N; ++I) {
    if (Params1->getParam(I)->getKind() != Params2->getParam(I)->getKind()) {
      if (Context.Complain) {
        Context.Diag2(Params2->getParam(I)->getLocation(), 
                      diag::err_odr_different_template_parameter_kind);
        Context.Diag1(Params1->getParam(I)->getLocation(),
                      diag::note_odr_template_parameter_here);
      }
      return false;
    }
    
    if (!Context.IsStructurallyEquivalent(Params1->getParam(I),
                                          Params2->getParam(I))) {
      
      return false;
    }
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     TemplateTypeParmDecl *D1,
                                     TemplateTypeParmDecl *D2) {
  if (D1->isParameterPack() != D2->isParameterPack()) {
    if (Context.Complain) {
      Context.Diag2(D2->getLocation(), diag::err_odr_parameter_pack_non_pack)
        << D2->isParameterPack();
      Context.Diag1(D1->getLocation(), diag::note_odr_parameter_pack_non_pack)
        << D1->isParameterPack();
    }
    return false;
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     NonTypeTemplateParmDecl *D1,
                                     NonTypeTemplateParmDecl *D2) {
  if (D1->isParameterPack() != D2->isParameterPack()) {
    if (Context.Complain) {
      Context.Diag2(D2->getLocation(), diag::err_odr_parameter_pack_non_pack)
        << D2->isParameterPack();
      Context.Diag1(D1->getLocation(), diag::note_odr_parameter_pack_non_pack)
        << D1->isParameterPack();
    }
    return false;
  }
  
  // Check types.
  if (!Context.IsStructurallyEquivalent(D1->getType(), D2->getType())) {
    if (Context.Complain) {
      Context.Diag2(D2->getLocation(),
                    diag::err_odr_non_type_parameter_type_inconsistent)
        << D2->getType() << D1->getType();
      Context.Diag1(D1->getLocation(), diag::note_odr_value_here)
        << D1->getType();
    }
    return false;
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     TemplateTemplateParmDecl *D1,
                                     TemplateTemplateParmDecl *D2) {
  if (D1->isParameterPack() != D2->isParameterPack()) {
    if (Context.Complain) {
      Context.Diag2(D2->getLocation(), diag::err_odr_parameter_pack_non_pack)
        << D2->isParameterPack();
      Context.Diag1(D1->getLocation(), diag::note_odr_parameter_pack_non_pack)
        << D1->isParameterPack();
    }
    return false;
  }

  // Check template parameter lists.
  return IsStructurallyEquivalent(Context, D1->getTemplateParameters(),
                                  D2->getTemplateParameters());
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     ClassTemplateDecl *D1, 
                                     ClassTemplateDecl *D2) {
  // Check template parameters.
  if (!IsStructurallyEquivalent(Context,
                                D1->getTemplateParameters(),
                                D2->getTemplateParameters()))
    return false;
  
  // Check the templated declaration.
  return Context.IsStructurallyEquivalent(D1->getTemplatedDecl(), 
                                          D2->getTemplatedDecl());
}

/// \brief Determine structural equivalence of two declarations.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     Decl *D1, Decl *D2) {
  // FIXME: Check for known structural equivalences via a callback of some sort.
  
  // Check whether we already know that these two declarations are not
  // structurally equivalent.
  if (Context.NonEquivalentDecls.count(std::make_pair(D1->getCanonicalDecl(),
                                                      D2->getCanonicalDecl())))
    return false;
  
  // Determine whether we've already produced a tentative equivalence for D1.
  Decl *&EquivToD1 = Context.TentativeEquivalences[D1->getCanonicalDecl()];
  if (EquivToD1)
    return EquivToD1 == D2->getCanonicalDecl();
  
  // Produce a tentative equivalence D1 <-> D2, which will be checked later.
  EquivToD1 = D2->getCanonicalDecl();
  Context.DeclsToCheck.push_back(D1->getCanonicalDecl());
  return true;
}

bool StructuralEquivalenceContext::IsStructurallyEquivalent(Decl *D1, 
                                                            Decl *D2) {
  if (!::IsStructurallyEquivalent(*this, D1, D2))
    return false;
  
  return !Finish();
}

bool StructuralEquivalenceContext::IsStructurallyEquivalent(QualType T1, 
                                                            QualType T2) {
  if (!::IsStructurallyEquivalent(*this, T1, T2))
    return false;
  
  return !Finish();
}

bool StructuralEquivalenceContext::Finish() {
  while (!DeclsToCheck.empty()) {
    // Check the next declaration.
    Decl *D1 = DeclsToCheck.front();
    DeclsToCheck.pop_front();
    
    Decl *D2 = TentativeEquivalences[D1];
    assert(D2 && "Unrecorded tentative equivalence?");
    
    bool Equivalent = true;
    
    // FIXME: Switch on all declaration kinds. For now, we're just going to
    // check the obvious ones.
    if (RecordDecl *Record1 = dyn_cast<RecordDecl>(D1)) {
      if (RecordDecl *Record2 = dyn_cast<RecordDecl>(D2)) {
        // Check for equivalent structure names.
        IdentifierInfo *Name1 = Record1->getIdentifier();
        if (!Name1 && Record1->getTypedefNameForAnonDecl())
          Name1 = Record1->getTypedefNameForAnonDecl()->getIdentifier();
        IdentifierInfo *Name2 = Record2->getIdentifier();
        if (!Name2 && Record2->getTypedefNameForAnonDecl())
          Name2 = Record2->getTypedefNameForAnonDecl()->getIdentifier();
        if (!::IsStructurallyEquivalent(Name1, Name2) ||
            !::IsStructurallyEquivalent(*this, Record1, Record2))
          Equivalent = false;
      } else {
        // Record/non-record mismatch.
        Equivalent = false;
      }
    } else if (EnumDecl *Enum1 = dyn_cast<EnumDecl>(D1)) {
      if (EnumDecl *Enum2 = dyn_cast<EnumDecl>(D2)) {
        // Check for equivalent enum names.
        IdentifierInfo *Name1 = Enum1->getIdentifier();
        if (!Name1 && Enum1->getTypedefNameForAnonDecl())
          Name1 = Enum1->getTypedefNameForAnonDecl()->getIdentifier();
        IdentifierInfo *Name2 = Enum2->getIdentifier();
        if (!Name2 && Enum2->getTypedefNameForAnonDecl())
          Name2 = Enum2->getTypedefNameForAnonDecl()->getIdentifier();
        if (!::IsStructurallyEquivalent(Name1, Name2) ||
            !::IsStructurallyEquivalent(*this, Enum1, Enum2))
          Equivalent = false;
      } else {
        // Enum/non-enum mismatch
        Equivalent = false;
      }
    } else if (TypedefNameDecl *Typedef1 = dyn_cast<TypedefNameDecl>(D1)) {
      if (TypedefNameDecl *Typedef2 = dyn_cast<TypedefNameDecl>(D2)) {
        if (!::IsStructurallyEquivalent(Typedef1->getIdentifier(),
                                        Typedef2->getIdentifier()) ||
            !::IsStructurallyEquivalent(*this,
                                        Typedef1->getUnderlyingType(),
                                        Typedef2->getUnderlyingType()))
          Equivalent = false;
      } else {
        // Typedef/non-typedef mismatch.
        Equivalent = false;
      }
    } else if (ClassTemplateDecl *ClassTemplate1 
                                           = dyn_cast<ClassTemplateDecl>(D1)) {
      if (ClassTemplateDecl *ClassTemplate2 = dyn_cast<ClassTemplateDecl>(D2)) {
        if (!::IsStructurallyEquivalent(ClassTemplate1->getIdentifier(),
                                        ClassTemplate2->getIdentifier()) ||
            !::IsStructurallyEquivalent(*this, ClassTemplate1, ClassTemplate2))
          Equivalent = false;
      } else {
        // Class template/non-class-template mismatch.
        Equivalent = false;
      }
    } else if (TemplateTypeParmDecl *TTP1= dyn_cast<TemplateTypeParmDecl>(D1)) {
      if (TemplateTypeParmDecl *TTP2 = dyn_cast<TemplateTypeParmDecl>(D2)) {
        if (!::IsStructurallyEquivalent(*this, TTP1, TTP2))
          Equivalent = false;
      } else {
        // Kind mismatch.
        Equivalent = false;
      }
    } else if (NonTypeTemplateParmDecl *NTTP1
                                     = dyn_cast<NonTypeTemplateParmDecl>(D1)) {
      if (NonTypeTemplateParmDecl *NTTP2
                                      = dyn_cast<NonTypeTemplateParmDecl>(D2)) {
        if (!::IsStructurallyEquivalent(*this, NTTP1, NTTP2))
          Equivalent = false;
      } else {
        // Kind mismatch.
        Equivalent = false;
      }
    } else if (TemplateTemplateParmDecl *TTP1
                                  = dyn_cast<TemplateTemplateParmDecl>(D1)) {
      if (TemplateTemplateParmDecl *TTP2
                                    = dyn_cast<TemplateTemplateParmDecl>(D2)) {
        if (!::IsStructurallyEquivalent(*this, TTP1, TTP2))
          Equivalent = false;
      } else {
        // Kind mismatch.
        Equivalent = false;
      }
    }
    
    if (!Equivalent) {
      // Note that these two declarations are not equivalent (and we already
      // know about it).
      NonEquivalentDecls.insert(std::make_pair(D1->getCanonicalDecl(),
                                               D2->getCanonicalDecl()));
      return true;
    }
    // FIXME: Check other declaration kinds!
  }
  
  return false;
}

//----------------------------------------------------------------------------
// Import Types
//----------------------------------------------------------------------------

QualType ASTNodeImporter::VisitType(const Type *T) {
  Importer.FromDiag(SourceLocation(), diag::err_unsupported_ast_node)
    << T->getTypeClassName();
  return QualType();
}

QualType ASTNodeImporter::VisitBuiltinType(const BuiltinType *T) {
  switch (T->getKind()) {
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
  QualType ToResultType = Importer.Import(T->getResultType());
  if (ToResultType.isNull())
    return QualType();

  return Importer.getToContext().getFunctionNoProtoType(ToResultType,
                                                        T->getExtInfo());
}

QualType ASTNodeImporter::VisitFunctionProtoType(const FunctionProtoType *T) {
  QualType ToResultType = Importer.Import(T->getResultType());
  if (ToResultType.isNull())
    return QualType();
  
  // Import argument types
  SmallVector<QualType, 4> ArgTypes;
  for (FunctionProtoType::arg_type_iterator A = T->arg_type_begin(),
                                         AEnd = T->arg_type_end();
       A != AEnd; ++A) {
    QualType ArgType = Importer.Import(*A);
    if (ArgType.isNull())
      return QualType();
    ArgTypes.push_back(ArgType);
  }
  
  // Import exception types
  SmallVector<QualType, 4> ExceptionTypes;
  for (FunctionProtoType::exception_iterator E = T->exception_begin(),
                                          EEnd = T->exception_end();
       E != EEnd; ++E) {
    QualType ExceptionType = Importer.Import(*E);
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
  ToEPI.NumExceptions = ExceptionTypes.size();
  ToEPI.Exceptions = ExceptionTypes.data();
  ToEPI.ConsumedArguments = FromEPI.ConsumedArguments;
  ToEPI.ExceptionSpecType = FromEPI.ExceptionSpecType;
  ToEPI.NoexceptExpr = Importer.Import(FromEPI.NoexceptExpr);
  ToEPI.ExceptionSpecDecl = cast_or_null<FunctionDecl>(
                                Importer.Import(FromEPI.ExceptionSpecDecl));
  ToEPI.ExceptionSpecTemplate = cast_or_null<FunctionDecl>(
                                Importer.Import(FromEPI.ExceptionSpecTemplate));

  return Importer.getToContext().getFunctionType(ToResultType, ArgTypes.data(),
                                                 ArgTypes.size(), ToEPI);
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
  // FIXME: Make sure that the "to" context supports C++0x!
  QualType FromDeduced = T->getDeducedType();
  QualType ToDeduced;
  if (!FromDeduced.isNull()) {
    ToDeduced = Importer.Import(FromDeduced);
    if (ToDeduced.isNull())
      return QualType();
  }
  
  return Importer.getToContext().getAutoType(ToDeduced);
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
                                                         ToTemplateArgs.data(), 
                                                         ToTemplateArgs.size(),
                                                               ToCanonType);
}

QualType ASTNodeImporter::VisitElaboratedType(const ElaboratedType *T) {
  NestedNameSpecifier *ToQualifier = 0;
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

  SmallVector<ObjCProtocolDecl *, 4> Protocols;
  for (ObjCObjectType::qual_iterator P = T->qual_begin(), 
                                     PEnd = T->qual_end();
       P != PEnd; ++P) {
    ObjCProtocolDecl *Protocol
      = dyn_cast_or_null<ObjCProtocolDecl>(Importer.Import(*P));
    if (!Protocol)
      return QualType();
    Protocols.push_back(Protocol);
  }

  return Importer.getToContext().getObjCObjectType(ToBaseType,
                                                   Protocols.data(),
                                                   Protocols.size());
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
      if (FromRecord->getDefinition() && !ToRecord->getDefinition()) {
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
  
  for (DeclContext::decl_iterator From = FromDC->decls_begin(),
                               FromEnd = FromDC->decls_end();
       From != FromEnd;
       ++From)
    Importer.Import(*From);
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
    ToData.UserDeclaredCopyConstructor = FromData.UserDeclaredCopyConstructor;
    ToData.UserDeclaredMoveConstructor = FromData.UserDeclaredMoveConstructor;
    ToData.UserDeclaredCopyAssignment = FromData.UserDeclaredCopyAssignment;
    ToData.UserDeclaredMoveAssignment = FromData.UserDeclaredMoveAssignment;
    ToData.UserDeclaredDestructor = FromData.UserDeclaredDestructor;
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
    ToData.HasOnlyCMembers = FromData.HasOnlyCMembers;
    ToData.HasInClassInitializer = FromData.HasInClassInitializer;
    ToData.HasTrivialDefaultConstructor = FromData.HasTrivialDefaultConstructor;
    ToData.HasConstexprNonCopyMoveConstructor
      = FromData.HasConstexprNonCopyMoveConstructor;
    ToData.DefaultedDefaultConstructorIsConstexpr
      = FromData.DefaultedDefaultConstructorIsConstexpr;
    ToData.HasConstexprDefaultConstructor
      = FromData.HasConstexprDefaultConstructor;
    ToData.HasTrivialCopyConstructor = FromData.HasTrivialCopyConstructor;
    ToData.HasTrivialMoveConstructor = FromData.HasTrivialMoveConstructor;
    ToData.HasTrivialCopyAssignment = FromData.HasTrivialCopyAssignment;
    ToData.HasTrivialMoveAssignment = FromData.HasTrivialMoveAssignment;
    ToData.HasTrivialDestructor = FromData.HasTrivialDestructor;
    ToData.HasIrrelevantDestructor = FromData.HasIrrelevantDestructor;
    ToData.HasNonLiteralTypeFieldsOrBases
      = FromData.HasNonLiteralTypeFieldsOrBases;
    // ComputedVisibleConversions not imported.
    ToData.UserProvidedDefaultConstructor
      = FromData.UserProvidedDefaultConstructor;
    ToData.DeclaredDefaultConstructor = FromData.DeclaredDefaultConstructor;
    ToData.DeclaredCopyConstructor = FromData.DeclaredCopyConstructor;
    ToData.DeclaredMoveConstructor = FromData.DeclaredMoveConstructor;
    ToData.DeclaredCopyAssignment = FromData.DeclaredCopyAssignment;
    ToData.DeclaredMoveAssignment = FromData.DeclaredMoveAssignment;
    ToData.DeclaredDestructor = FromData.DeclaredDestructor;
    ToData.FailedImplicitMoveConstructor
      = FromData.FailedImplicitMoveConstructor;
    ToData.FailedImplicitMoveAssignment = FromData.FailedImplicitMoveAssignment;
    ToData.IsLambda = FromData.IsLambda;

    SmallVector<CXXBaseSpecifier *, 4> Bases;
    for (CXXRecordDecl::base_class_iterator 
                  Base1 = FromCXX->bases_begin(),
            FromBaseEnd = FromCXX->bases_end();
         Base1 != FromBaseEnd;
         ++Base1) {
      QualType T = Importer.Import(Base1->getType());
      if (T.isNull())
        return true;

      SourceLocation EllipsisLoc;
      if (Base1->isPackExpansion())
        EllipsisLoc = Importer.Import(Base1->getEllipsisLoc());

      // Ensure that we have a definition for the base.
      ImportDefinitionIfNeeded(Base1->getType()->getAsCXXRecordDecl());
        
      Bases.push_back(
                    new (Importer.getToContext()) 
                      CXXBaseSpecifier(Importer.Import(Base1->getSourceRange()),
                                       Base1->isVirtual(),
                                       Base1->isBaseOfClass(),
                                       Base1->getAccessSpecifierAsWritten(),
                                   Importer.Import(Base1->getTypeSourceInfo()),
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
  SmallVector<NamedDecl *, 4> ToParams;
  ToParams.reserve(Params->size());
  for (TemplateParameterList::iterator P = Params->begin(), 
                                    PEnd = Params->end();
       P != PEnd; ++P) {
    Decl *To = Importer.Import(*P);
    if (!To)
      return 0;
    
    ToParams.push_back(cast<NamedDecl>(To));
  }
  
  return TemplateParameterList::Create(Importer.getToContext(),
                                       Importer.Import(Params->getTemplateLoc()),
                                       Importer.Import(Params->getLAngleLoc()),
                                       ToParams.data(), ToParams.size(),
                                       Importer.Import(Params->getRAngleLoc()));
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
    ValueDecl *FromD = From.getAsDecl();
    if (ValueDecl *To = cast_or_null<ValueDecl>(Importer.Import(FromD)))
      return TemplateArgument(To, From.isDeclForReferenceParam());
    return TemplateArgument();
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
    
    TemplateArgument *ToArgs 
      = new (Importer.getToContext()) TemplateArgument[ToPack.size()];
    std::copy(ToPack.begin(), ToPack.end(), ToArgs);
    return TemplateArgument(ToArgs, ToPack.size());
  }
  }
  
  llvm_unreachable("Invalid template argument kind");
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
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls(),
                                   false, Complain);
  return Ctx.IsStructurallyEquivalent(FromRecord, ToRecord);
}

bool ASTNodeImporter::IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToEnum) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(FromEnum, ToEnum);
}

bool ASTNodeImporter::IsStructuralMatch(ClassTemplateDecl *From, 
                                        ClassTemplateDecl *To) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(From, To);  
}

Decl *ASTNodeImporter::VisitDecl(Decl *D) {
  Importer.FromDiag(D->getLocation(), diag::err_unsupported_ast_node)
    << D->getDeclKindName();
  return 0;
}

Decl *ASTNodeImporter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  TranslationUnitDecl *ToD = 
    Importer.getToContext().getTranslationUnitDecl();
    
  Importer.Imported(D, ToD);
    
  return ToD;
}

Decl *ASTNodeImporter::VisitNamespaceDecl(NamespaceDecl *D) {
  // Import the major distinguishing characteristics of this namespace.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  NamespaceDecl *MergeWithNamespace = 0;
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
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(Name, FoundDecls);
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
                                        /*PrevDecl=*/0);
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
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // If this typedef is not in block scope, determine whether we've
  // seen a typedef with the same name (that we can merge with) or any
  // other entity by that name (which name lookup could conflict with).
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(Name, FoundDecls);
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
        return 0;
    }
  }
  
  // Import the underlying type of this typedef;
  QualType T = Importer.Import(D->getUnderlyingType());
  if (T.isNull())
    return 0;
  
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

Decl *ASTNodeImporter::VisitEnumDecl(EnumDecl *D) {
  // Import the major distinguishing characteristics of this enum.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
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
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(SearchName, FoundDecls);
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
                                  Loc, Name.getAsIdentifierInfo(), 0,
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
    return 0;
  D2->setIntegerType(ToIntegerType);
  
  // Import the definition
  if (D->isCompleteDefinition() && ImportDefinition(D, D2))
    return 0;

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
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
      
  // Figure out what structure name we're looking for.
  unsigned IDNS = Decl::IDNS_Tag;
  DeclarationName SearchName = Name;
  if (!SearchName && D->getTypedefNameForAnonDecl()) {
    SearchName = Importer.Import(D->getTypedefNameForAnonDecl()->getDeclName());
    IDNS = Decl::IDNS_Ordinary;
  } else if (Importer.getToContext().getLangOpts().CPlusPlus)
    IDNS |= Decl::IDNS_Ordinary;

  // We may already have a record of the same name; try to find and match it.
  RecordDecl *AdoptDecl = 0;
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(SearchName, FoundDecls);
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
          if (llvm::Optional<unsigned> Index1
              = findAnonymousStructOrUnionIndex(D)) {
            if (llvm::Optional<unsigned> Index2
                = findAnonymousStructOrUnionIndex(FoundRecord)) {
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
    if (isa<CXXRecordDecl>(D)) {
      CXXRecordDecl *D2CXX = CXXRecordDecl::Create(Importer.getToContext(), 
                                                   D->getTagKind(),
                                                   DC, StartLoc, Loc,
                                                   Name.getAsIdentifierInfo());
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
    return 0;
  
  return D2;
}

Decl *ASTNodeImporter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  // Import the major distinguishing characteristics of this enumerator.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;

  // Determine whether there are any other declarations with the same name and 
  // in the same context.
  if (!LexicalDC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;
      
      ConflictingDecls.push_back(FoundDecls[I]);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return 0;
    }
  }
  
  Expr *Init = Importer.Import(D->getInitExpr());
  if (D->getInitExpr() && !Init)
    return 0;
  
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
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Try to find a function in our own ("to") context with the same name, same
  // type, and in the same context as the function we're importing.
  if (!LexicalDC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;
    
      if (FunctionDecl *FoundFunction = dyn_cast<FunctionDecl>(FoundDecls[I])) {
        if (isExternalLinkage(FoundFunction->getLinkage()) &&
            isExternalLinkage(D->getLinkage())) {
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
        return 0;
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
    if (FromEPI.ExceptionSpecDecl || FromEPI.ExceptionSpecTemplate ||
        FromEPI.NoexceptExpr) {
      FunctionProtoType::ExtProtoInfo DefaultEPI;
      FromTy = Importer.getFromContext().getFunctionType(
                            FromFPT->getResultType(),
                            FromFPT->arg_type_begin(),
                            FromFPT->arg_type_end() - FromFPT->arg_type_begin(),
                            DefaultEPI);
      usedDifferentExceptionSpec = true;
    }
  }

  // Import the type.
  QualType T = Importer.Import(FromTy);
  if (T.isNull())
    return 0;
  
  // Import the function parameters.
  SmallVector<ParmVarDecl *, 8> Parameters;
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P) {
    ParmVarDecl *ToP = cast_or_null<ParmVarDecl>(Importer.Import(*P));
    if (!ToP)
      return 0;
    
    Parameters.push_back(ToP);
  }
  
  // Create the imported function.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  FunctionDecl *ToFunction = 0;
  if (CXXConstructorDecl *FromConstructor = dyn_cast<CXXConstructorDecl>(D)) {
    ToFunction = CXXConstructorDecl::Create(Importer.getToContext(),
                                            cast<CXXRecordDecl>(DC),
                                            D->getInnerLocStart(),
                                            NameInfo, T, TInfo, 
                                            FromConstructor->isExplicit(),
                                            D->isInlineSpecified(), 
                                            D->isImplicit(),
                                            D->isConstexpr());
  } else if (isa<CXXDestructorDecl>(D)) {
    ToFunction = CXXDestructorDecl::Create(Importer.getToContext(),
                                           cast<CXXRecordDecl>(DC),
                                           D->getInnerLocStart(),
                                           NameInfo, T, TInfo,
                                           D->isInlineSpecified(),
                                           D->isImplicit());
  } else if (CXXConversionDecl *FromConversion
                                           = dyn_cast<CXXConversionDecl>(D)) {
    ToFunction = CXXConversionDecl::Create(Importer.getToContext(), 
                                           cast<CXXRecordDecl>(DC),
                                           D->getInnerLocStart(),
                                           NameInfo, T, TInfo,
                                           D->isInlineSpecified(),
                                           FromConversion->isExplicit(),
                                           D->isConstexpr(),
                                           Importer.Import(D->getLocEnd()));
  } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    ToFunction = CXXMethodDecl::Create(Importer.getToContext(), 
                                       cast<CXXRecordDecl>(DC),
                                       D->getInnerLocStart(),
                                       NameInfo, T, TInfo,
                                       Method->isStatic(),
                                       Method->getStorageClassAsWritten(),
                                       Method->isInlineSpecified(),
                                       D->isConstexpr(),
                                       Importer.Import(D->getLocEnd()));
  } else {
    ToFunction = FunctionDecl::Create(Importer.getToContext(), DC,
                                      D->getInnerLocStart(),
                                      NameInfo, T, TInfo, D->getStorageClass(),
                                      D->getStorageClassAsWritten(),
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
      return 0;
    ToFunction->setType(T);
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
  for (DeclContext::decl_iterator D = Owner->noload_decls_begin(),
                               DEnd = Owner->noload_decls_end();
       D != DEnd; ++D) {
    if (*D == F)
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
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // Determine whether we've already imported this field. 
  llvm::SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
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
      return 0;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  Expr *BitWidth = Importer.Import(D->getBitWidth());
  if (!BitWidth && D->getBitWidth())
    return 0;
  
  FieldDecl *ToField = FieldDecl::Create(Importer.getToContext(), DC,
                                         Importer.Import(D->getInnerLocStart()),
                                         Loc, Name.getAsIdentifierInfo(),
                                         T, TInfo, BitWidth, D->isMutable(),
                                         D->getInClassInitStyle());
  ToField->setAccess(D->getAccess());
  ToField->setLexicalDeclContext(LexicalDC);
  if (ToField->hasInClassInitializer())
    ToField->setInClassInitializer(D->getInClassInitializer());
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
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Determine whether we've already imported this field. 
  llvm::SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (IndirectFieldDecl *FoundField 
                                = dyn_cast<IndirectFieldDecl>(FoundDecls[I])) {
      // For anonymous indirect fields, match up by index.
      if (!Name && getFieldIndex(D) != getFieldIndex(FoundField))
        continue;

      if (Importer.IsStructurallyEquivalent(D->getType(), 
                                            FoundField->getType(),
                                            Name)) {
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
      return 0;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;

  NamedDecl **NamedChain =
    new (Importer.getToContext())NamedDecl*[D->getChainingSize()];

  unsigned i = 0;
  for (IndirectFieldDecl::chain_iterator PI = D->chain_begin(),
       PE = D->chain_end(); PI != PE; ++PI) {
    Decl* D = Importer.Import(*PI);
    if (!D)
      return 0;
    NamedChain[i++] = cast<NamedDecl>(D);
  }

  IndirectFieldDecl *ToIndirectField = IndirectFieldDecl::Create(
                                         Importer.getToContext(), DC,
                                         Loc, Name.getAsIdentifierInfo(), T,
                                         NamedChain, D->getChainingSize());
  ToIndirectField->setAccess(D->getAccess());
  ToIndirectField->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToIndirectField);
  LexicalDC->addDeclInternal(ToIndirectField);
  return ToIndirectField;
}

Decl *ASTNodeImporter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  // Import the major distinguishing characteristics of an ivar.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // Determine whether we've already imported this ivar 
  llvm::SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
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
      return 0;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  Expr *BitWidth = Importer.Import(D->getBitWidth());
  if (!BitWidth && D->getBitWidth())
    return 0;
  
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
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // Try to find a variable in our own ("to") context with the same name and
  // in the same context as the variable we're importing.
  if (D->isFileVarDecl()) {
    VarDecl *MergeWithVar = 0;
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(Name, FoundDecls);
    for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
      if (!FoundDecls[I]->isInIdentifierNamespace(IDNS))
        continue;
      
      if (VarDecl *FoundVar = dyn_cast<VarDecl>(FoundDecls[I])) {
        // We have found a variable that we may need to merge with. Check it.
        if (isExternalLinkage(FoundVar->getLinkage()) &&
            isExternalLinkage(D->getLinkage())) {
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
                return 0;
              
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
        return 0;
    }
  }
    
  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Create the imported variable.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  VarDecl *ToVar = VarDecl::Create(Importer.getToContext(), DC,
                                   Importer.Import(D->getInnerLocStart()),
                                   Loc, Name.getAsIdentifierInfo(),
                                   T, TInfo,
                                   D->getStorageClass(),
                                   D->getStorageClassAsWritten());
  ToVar->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  ToVar->setAccess(D->getAccess());
  ToVar->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToVar);
  LexicalDC->addDeclInternal(ToVar);

  // Merge the initializer.
  // FIXME: Can we really import any initializer? Alternatively, we could force
  // ourselves to import every declaration of a variable and then only use
  // getInit() here.
  ToVar->setInit(Importer.Import(const_cast<Expr *>(D->getAnyInitializer())));

  // FIXME: Other bits to merge?
  
  return ToVar;
}

Decl *ASTNodeImporter::VisitImplicitParamDecl(ImplicitParamDecl *D) {
  // Parameters are created in the translation unit's context, then moved
  // into the function declaration's context afterward.
  DeclContext *DC = Importer.getToContext().getTranslationUnitDecl();
  
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import the parameter's type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
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
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import the parameter's type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Create the imported parameter.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  ParmVarDecl *ToParm = ParmVarDecl::Create(Importer.getToContext(), DC,
                                     Importer.Import(D->getInnerLocStart()),
                                            Loc, Name.getAsIdentifierInfo(),
                                            T, TInfo, D->getStorageClass(),
                                             D->getStorageClassAsWritten(),
                                            /*FIXME: Default argument*/ 0);
  ToParm->setHasInheritedDefaultArg(D->hasInheritedDefaultArg());
  return Importer.Imported(D, ToParm);
}

Decl *ASTNodeImporter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  // Import the major distinguishing characteristics of a method.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  llvm::SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (ObjCMethodDecl *FoundMethod = dyn_cast<ObjCMethodDecl>(FoundDecls[I])) {
      if (FoundMethod->isInstanceMethod() != D->isInstanceMethod())
        continue;

      // Check return types.
      if (!Importer.IsStructurallyEquivalent(D->getResultType(),
                                             FoundMethod->getResultType())) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_result_type_inconsistent)
          << D->isInstanceMethod() << Name
          << D->getResultType() << FoundMethod->getResultType();
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return 0;
      }

      // Check the number of parameters.
      if (D->param_size() != FoundMethod->param_size()) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_num_params_inconsistent)
          << D->isInstanceMethod() << Name
          << D->param_size() << FoundMethod->param_size();
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return 0;
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
          return 0;
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
        return 0;
      }

      // FIXME: Any other bits we need to merge?
      return Importer.Imported(D, FoundMethod);
    }
  }

  // Import the result type.
  QualType ResultTy = Importer.Import(D->getResultType());
  if (ResultTy.isNull())
    return 0;

  TypeSourceInfo *ResultTInfo = Importer.Import(D->getResultTypeSourceInfo());

  ObjCMethodDecl *ToMethod
    = ObjCMethodDecl::Create(Importer.getToContext(),
                             Loc,
                             Importer.Import(D->getLocEnd()),
                             Name.getObjCSelector(),
                             ResultTy, ResultTInfo, DC,
                             D->isInstanceMethod(),
                             D->isVariadic(),
                             D->isPropertyAccessor(),
                             D->isImplicit(),
                             D->isDefined(),
                             D->getImplementationControl(),
                             D->hasRelatedResultType());

  // FIXME: When we decide to merge method definitions, we'll need to
  // deal with implicit parameters.

  // Import the parameters
  SmallVector<ParmVarDecl *, 5> ToParams;
  for (ObjCMethodDecl::param_iterator FromP = D->param_begin(),
                                   FromPEnd = D->param_end();
       FromP != FromPEnd; 
       ++FromP) {
    ParmVarDecl *ToP = cast_or_null<ParmVarDecl>(Importer.Import(*FromP));
    if (!ToP)
      return 0;
    
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

Decl *ASTNodeImporter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  // Import the major distinguishing characteristics of a category.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  ObjCInterfaceDecl *ToInterface
    = cast_or_null<ObjCInterfaceDecl>(Importer.Import(D->getClassInterface()));
  if (!ToInterface)
    return 0;
  
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
                                       Importer.Import(D->getIvarLBraceLoc()),
                                       Importer.Import(D->getIvarRBraceLoc()));
    ToCategory->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToCategory);
    Importer.Imported(D, ToCategory);
    
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
        return 0;
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
      return 0;
    
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
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of a protocol.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  ObjCProtocolDecl *MergeWithProtocol = 0;
  llvm::SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
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
                                       /*PrevDecl=*/0);
    ToProto->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToProto);
  }
    
  Importer.Imported(D, ToProto);

  if (D->isThisDeclarationADefinition() && ImportDefinition(D, ToProto))
    return 0;
  
  return ToProto;
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
    ObjCInterfaceDecl *Super = cast_or_null<ObjCInterfaceDecl>(
                                 Importer.Import(From->getSuperClass()));
    if (!Super)
      return true;
    
    To->setSuperClass(Super);
    To->setSuperClassLoc(Importer.Import(From->getSuperClassLoc()));
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
  for (ObjCCategoryDecl *FromCat = From->getCategoryList(); FromCat;
       FromCat = FromCat->getNextClassCategory())
    Importer.Import(FromCat);

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

Decl *ASTNodeImporter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // If this class has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  ObjCInterfaceDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of an @interface.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Look for an existing interface with the same name.
  ObjCInterfaceDecl *MergeWithIface = 0;
  llvm::SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
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
                                        /*PrevDecl=*/0,Loc,
                                        D->isImplicitInterfaceDecl());
    ToIface->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToIface);
  }
  Importer.Imported(D, ToIface);
  
  if (D->isThisDeclarationADefinition() && ImportDefinition(D, ToIface))
    return 0;
    
  return ToIface;
}

Decl *ASTNodeImporter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  ObjCCategoryDecl *Category = cast_or_null<ObjCCategoryDecl>(
                                        Importer.Import(D->getCategoryDecl()));
  if (!Category)
    return 0;
  
  ObjCCategoryImplDecl *ToImpl = Category->getImplementation();
  if (!ToImpl) {
    DeclContext *DC = Importer.ImportContext(D->getDeclContext());
    if (!DC)
      return 0;
    
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
        return 0;
      
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
    return 0;

  // Import the superclass, if any.
  ObjCInterfaceDecl *Super = 0;
  if (D->getSuperClass()) {
    Super = cast_or_null<ObjCInterfaceDecl>(
                                          Importer.Import(D->getSuperClass()));
    if (!Super)
      return 0;
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
                                          Importer.Import(D->getIvarLBraceLoc()),
                                          Importer.Import(D->getIvarRBraceLoc()));
    
    if (D->getDeclContext() != D->getLexicalDeclContext()) {
      DeclContext *LexicalDC
        = Importer.ImportContext(D->getLexicalDeclContext());
      if (!LexicalDC)
        return 0;
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
         !declaresSameEntity(Super->getCanonicalDecl(), Impl->getSuperClass()))) {
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
      return 0;
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
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Check whether we have already imported this property.
  llvm::SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
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
        return 0;
      }

      // FIXME: Check property attributes, getters, setters, etc.?

      // Consider these properties to be equivalent.
      Importer.Imported(D, FoundProp);
      return FoundProp;
    }
  }

  // Import the type.
  TypeSourceInfo *T = Importer.Import(D->getTypeSourceInfo());
  if (!T)
    return 0;

  // Create the new property.
  ObjCPropertyDecl *ToProperty
    = ObjCPropertyDecl::Create(Importer.getToContext(), DC, Loc,
                               Name.getAsIdentifierInfo(), 
                               Importer.Import(D->getAtLoc()),
                               Importer.Import(D->getLParenLoc()),
                               T,
                               D->getPropertyImplementation());
  Importer.Imported(D, ToProperty);
  ToProperty->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToProperty);

  ToProperty->setPropertyAttributes(D->getPropertyAttributes());
  ToProperty->setPropertyAttributesAsWritten(
                                      D->getPropertyAttributesAsWritten());
  ToProperty->setGetterName(Importer.Import(D->getGetterName()));
  ToProperty->setSetterName(Importer.Import(D->getSetterName()));
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
    return 0;

  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return 0;
  
  // Import the lexical declaration context.
  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return 0;
  }

  ObjCImplDecl *InImpl = dyn_cast<ObjCImplDecl>(LexicalDC);
  if (!InImpl)
    return 0;

  // Import the ivar (for an @synthesize).
  ObjCIvarDecl *Ivar = 0;
  if (D->getPropertyIvarDecl()) {
    Ivar = cast_or_null<ObjCIvarDecl>(
                                    Importer.Import(D->getPropertyIvarDecl()));
    if (!Ivar)
      return 0;
  }

  ObjCPropertyImplDecl *ToImpl
    = InImpl->FindPropertyImplDecl(Property->getIdentifier());
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
      return 0;
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
      return 0;
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
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());

  // Import the type of this declaration.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Import type-source information.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  if (D->getTypeSourceInfo() && !TInfo)
    return 0;
  
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
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import template parameters.
  TemplateParameterList *TemplateParams
    = ImportTemplateParameterList(D->getTemplateParameters());
  if (!TemplateParams)
    return 0;
  
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
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this class template.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // We may already have a template of the same name; try to find and match it.
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    llvm::SmallVector<NamedDecl *, 2> FoundDecls;
    DC->localUncachedLookup(Name, FoundDecls);
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
      return 0;
  }

  CXXRecordDecl *DTemplated = D->getTemplatedDecl();
  
  // Create the declaration that is being templated.
  SourceLocation StartLoc = Importer.Import(DTemplated->getLocStart());
  SourceLocation IdLoc = Importer.Import(DTemplated->getLocation());
  CXXRecordDecl *D2Templated = CXXRecordDecl::Create(Importer.getToContext(),
                                                     DTemplated->getTagKind(),
                                                     DC, StartLoc, IdLoc,
                                                   Name.getAsIdentifierInfo());
  D2Templated->setAccess(DTemplated->getAccess());
  D2Templated->setQualifierInfo(Importer.Import(DTemplated->getQualifierLoc()));
  D2Templated->setLexicalDeclContext(LexicalDC);
  
  // Create the class template declaration itself.
  TemplateParameterList *TemplateParams
    = ImportTemplateParameterList(D->getTemplateParameters());
  if (!TemplateParams)
    return 0;
  
  ClassTemplateDecl *D2 = ClassTemplateDecl::Create(Importer.getToContext(), DC, 
                                                    Loc, Name, TemplateParams, 
                                                    D2Templated, 
  /*PrevDecl=*/0);
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
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }

  ClassTemplateDecl *ClassTemplate
    = cast_or_null<ClassTemplateDecl>(Importer.Import(
                                                 D->getSpecializedTemplate()));
  if (!ClassTemplate)
    return 0;
  
  // Import the context of this declaration.
  DeclContext *DC = ClassTemplate->getDeclContext();
  if (!DC)
    return 0;
  
  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return 0;
  }
  
  // Import the location of this declaration.
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  SourceLocation IdLoc = Importer.Import(D->getLocation());

  // Import template arguments.
  SmallVector<TemplateArgument, 2> TemplateArgs;
  if (ImportTemplateArguments(D->getTemplateArgs().data(), 
                              D->getTemplateArgs().size(),
                              TemplateArgs))
    return 0;
  
  // Try to find an existing specialization with these template arguments.
  void *InsertPos = 0;
  ClassTemplateSpecializationDecl *D2
    = ClassTemplate->findSpecialization(TemplateArgs.data(), 
                                        TemplateArgs.size(), InsertPos);
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
    D2 = ClassTemplateSpecializationDecl::Create(Importer.getToContext(), 
                                                 D->getTagKind(), DC, 
                                                 StartLoc, IdLoc,
                                                 ClassTemplate,
                                                 TemplateArgs.data(), 
                                                 TemplateArgs.size(), 
                                                 /*PrevDecl=*/0);
    D2->setSpecializationKind(D->getSpecializationKind());

    // Add this specialization to the class template.
    ClassTemplate->AddSpecialization(D2, InsertPos);
    
    // Import the qualifier, if any.
    D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
    
    // Add the specialization to this context.
    D2->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(D2);
  }
  Importer.Imported(D, D2);
  
  if (D->isCompleteDefinition() && ImportDefinition(D, D2))
    return 0;
  
  return D2;
}

//----------------------------------------------------------------------------
// Import Statements
//----------------------------------------------------------------------------

Stmt *ASTNodeImporter::VisitStmt(Stmt *S) {
  Importer.FromDiag(S->getLocStart(), diag::err_unsupported_ast_node)
    << S->getStmtClassName();
  return 0;
}

//----------------------------------------------------------------------------
// Import Expressions
//----------------------------------------------------------------------------
Expr *ASTNodeImporter::VisitExpr(Expr *E) {
  Importer.FromDiag(E->getLocStart(), diag::err_unsupported_ast_node)
    << E->getStmtClassName();
  return 0;
}

Expr *ASTNodeImporter::VisitDeclRefExpr(DeclRefExpr *E) {
  ValueDecl *ToD = cast_or_null<ValueDecl>(Importer.Import(E->getDecl()));
  if (!ToD)
    return 0;

  NamedDecl *FoundD = 0;
  if (E->getDecl() != E->getFoundDecl()) {
    FoundD = cast_or_null<NamedDecl>(Importer.Import(E->getFoundDecl()));
    if (!FoundD)
      return 0;
  }
  
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  DeclRefExpr *DRE = DeclRefExpr::Create(Importer.getToContext(), 
                                         Importer.Import(E->getQualifierLoc()),
                                   Importer.Import(E->getTemplateKeywordLoc()),
                                         ToD,
                                         E->refersToEnclosingLocal(),
                                         Importer.Import(E->getLocation()),
                                         T, E->getValueKind(),
                                         FoundD,
                                         /*FIXME:TemplateArgs=*/0);
  if (E->hadMultipleCandidates())
    DRE->setHadMultipleCandidates(true);
  return DRE;
}

Expr *ASTNodeImporter::VisitIntegerLiteral(IntegerLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  return IntegerLiteral::Create(Importer.getToContext(), 
                                E->getValue(), T,
                                Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitCharacterLiteral(CharacterLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;
  
  return new (Importer.getToContext()) CharacterLiteral(E->getValue(),
                                                        E->getKind(), T,
                                          Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitParenExpr(ParenExpr *E) {
  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;
  
  return new (Importer.getToContext()) 
                                  ParenExpr(Importer.Import(E->getLParen()),
                                            Importer.Import(E->getRParen()),
                                            SubExpr);
}

Expr *ASTNodeImporter::VisitUnaryOperator(UnaryOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;
  
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
      return 0;
    
    return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(E->getKind(),
                                           TInfo, ResultType,
                                           Importer.Import(E->getOperatorLoc()),
                                           Importer.Import(E->getRParenLoc()));
  }
  
  Expr *SubExpr = Importer.Import(E->getArgumentExpr());
  if (!SubExpr)
    return 0;
  
  return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(E->getKind(),
                                          SubExpr, ResultType,
                                          Importer.Import(E->getOperatorLoc()),
                                          Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitBinaryOperator(BinaryOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  Expr *LHS = Importer.Import(E->getLHS());
  if (!LHS)
    return 0;
  
  Expr *RHS = Importer.Import(E->getRHS());
  if (!RHS)
    return 0;
  
  return new (Importer.getToContext()) BinaryOperator(LHS, RHS, E->getOpcode(),
                                                      T, E->getValueKind(),
                                                      E->getObjectKind(),
                                           Importer.Import(E->getOperatorLoc()),
                                                      E->isFPContractable());
}

Expr *ASTNodeImporter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;
  
  QualType CompLHSType = Importer.Import(E->getComputationLHSType());
  if (CompLHSType.isNull())
    return 0;
  
  QualType CompResultType = Importer.Import(E->getComputationResultType());
  if (CompResultType.isNull())
    return 0;
  
  Expr *LHS = Importer.Import(E->getLHS());
  if (!LHS)
    return 0;
  
  Expr *RHS = Importer.Import(E->getRHS());
  if (!RHS)
    return 0;
  
  return new (Importer.getToContext()) 
                        CompoundAssignOperator(LHS, RHS, E->getOpcode(),
                                               T, E->getValueKind(),
                                               E->getObjectKind(),
                                               CompLHSType, CompResultType,
                                           Importer.Import(E->getOperatorLoc()),
                                               E->isFPContractable());
}

static bool ImportCastPath(CastExpr *E, CXXCastPath &Path) {
  if (E->path_empty()) return false;

  // TODO: import cast paths
  return true;
}

Expr *ASTNodeImporter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;

  CXXCastPath BasePath;
  if (ImportCastPath(E, BasePath))
    return 0;

  return ImplicitCastExpr::Create(Importer.getToContext(), T, E->getCastKind(),
                                  SubExpr, &BasePath, E->getValueKind());
}

Expr *ASTNodeImporter::VisitCStyleCastExpr(CStyleCastExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;
  
  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;

  TypeSourceInfo *TInfo = Importer.Import(E->getTypeInfoAsWritten());
  if (!TInfo && E->getTypeInfoAsWritten())
    return 0;
  
  CXXCastPath BasePath;
  if (ImportCastPath(E, BasePath))
    return 0;

  return CStyleCastExpr::Create(Importer.getToContext(), T,
                                E->getValueKind(), E->getCastKind(),
                                SubExpr, &BasePath, TInfo,
                                Importer.Import(E->getLParenLoc()),
                                Importer.Import(E->getRParenLoc()));
}

ASTImporter::ASTImporter(ASTContext &ToContext, FileManager &ToFileManager,
                         ASTContext &FromContext, FileManager &FromFileManager,
                         bool MinimalImport)
  : ToContext(ToContext), FromContext(FromContext),
    ToFileManager(ToFileManager), FromFileManager(FromFileManager),
    Minimal(MinimalImport) 
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
    return 0;

  return ToContext.getTrivialTypeSourceInfo(T, 
                        FromTSI->getTypeLoc().getLocStart());
}

Decl *ASTImporter::Import(Decl *FromD) {
  if (!FromD)
    return 0;

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
    return 0;
  
  // Record the imported declaration.
  ImportedDecls[FromD] = ToD;
  
  if (TagDecl *FromTag = dyn_cast<TagDecl>(FromD)) {
    // Keep track of anonymous tags that have an associated typedef.
    if (FromTag->getTypedefNameForAnonDecl())
      AnonTagsWithPendingTypedefs.push_back(FromTag);
  } else if (TypedefNameDecl *FromTypedef = dyn_cast<TypedefNameDecl>(FromD)) {
    // When we've finished transforming a typedef, see whether it was the
    // typedef for an anonymous tag.
    for (SmallVector<TagDecl *, 4>::iterator
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
    return 0;
  
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
    return 0;

  return cast_or_null<Expr>(Import(cast<Stmt>(FromE)));
}

Stmt *ASTImporter::Import(Stmt *FromS) {
  if (!FromS)
    return 0;

  // Check whether we've already imported this declaration.  
  llvm::DenseMap<Stmt *, Stmt *>::iterator Pos = ImportedStmts.find(FromS);
  if (Pos != ImportedStmts.end())
    return Pos->second;
  
  // Import the type
  ASTNodeImporter Importer(*this);
  Stmt *ToS = Importer.Visit(FromS);
  if (!ToS)
    return 0;
  
  // Record the imported declaration.
  ImportedStmts[FromS] = ToS;
  return ToS;
}

NestedNameSpecifier *ASTImporter::Import(NestedNameSpecifier *FromNNS) {
  if (!FromNNS)
    return 0;

  NestedNameSpecifier *prefix = Import(FromNNS->getPrefix());

  switch (FromNNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    if (IdentifierInfo *II = Import(FromNNS->getAsIdentifier())) {
      return NestedNameSpecifier::Create(ToContext, prefix, II);
    }
    return 0;

  case NestedNameSpecifier::Namespace:
    if (NamespaceDecl *NS = 
          cast<NamespaceDecl>(Import(FromNNS->getAsNamespace()))) {
      return NestedNameSpecifier::Create(ToContext, prefix, NS);
    }
    return 0;

  case NestedNameSpecifier::NamespaceAlias:
    if (NamespaceAliasDecl *NSAD = 
          cast<NamespaceAliasDecl>(Import(FromNNS->getAsNamespaceAlias()))) {
      return NestedNameSpecifier::Create(ToContext, prefix, NSAD);
    }
    return 0;

  case NestedNameSpecifier::Global:
    return NestedNameSpecifier::GlobalSpecifier(ToContext);

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
    return 0;
  }

  llvm_unreachable("Invalid nested name specifier kind");
}

NestedNameSpecifierLoc ASTImporter::Import(NestedNameSpecifierLoc FromNNS) {
  // FIXME: Implement!
  return NestedNameSpecifierLoc();
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
  
  // For now, map everything down to its spelling location, so that we
  // don't have to import macro expansions.
  // FIXME: Import macro expansions!
  FromLoc = FromSM.getSpellingLoc(FromLoc);
  std::pair<FileID, unsigned> Decomposed = FromSM.getDecomposedLoc(FromLoc);
  SourceManager &ToSM = ToContext.getSourceManager();
  return ToSM.getLocForStartOfFile(Import(Decomposed.first))
             .getLocWithOffset(Decomposed.second);
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
  if (Cache->OrigEntry) {
    // FIXME: We probably want to use getVirtualFile(), so we don't hit the
    // disk again
    // FIXME: We definitely want to re-use the existing MemoryBuffer, rather
    // than mmap the files several times.
    const FileEntry *Entry = ToFileManager.getFile(Cache->OrigEntry->getName());
    ToID = ToSM.createFileID(Entry, ToIncludeLoc, 
                             FromSLoc.getFile().getFileCharacteristic());
  } else {
    // FIXME: We want to re-use the existing MemoryBuffer!
    const llvm::MemoryBuffer *
        FromBuf = Cache->getBuffer(FromContext.getDiagnostics(), FromSM);
    llvm::MemoryBuffer *ToBuf
      = llvm::MemoryBuffer::getMemBufferCopy(FromBuf->getBuffer(),
                                             FromBuf->getBufferIdentifier());
    ToID = ToSM.createFileIDForMemBuffer(ToBuf,
                                    FromSLoc.getFile().getFileCharacteristic());
  }
  
  
  ImportedFileIDs[FromID] = ToID;
  return ToID;
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
    return 0;

  return &ToContext.Idents.get(FromId->getName());
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
  return ToContext.getDiagnostics().Report(Loc, DiagID);
}

DiagnosticBuilder ASTImporter::FromDiag(SourceLocation Loc, unsigned DiagID) {
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
