//===-- ODRHash.cpp - Hashing to diagnose ODR failures ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ODRHash class, which calculates a hash based
/// on AST nodes, which is stable across different runs.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ODRHash.h"

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeVisitor.h"

using namespace clang;


// Hashing for Stmt is with Stmt::Profile, since they derive from the same base
// class.
void ODRHash::AddStmt(const Stmt *S) {
  assert(S && "Expecting non-null pointer.");
  S->ProcessODRHash(ID, *this);
}

void ODRHash::AddIdentifierInfo(const IdentifierInfo *II) {
  assert(II && "Expecting non-null pointer.");
  ID.AddString(II->getName());
}

void ODRHash::AddNestedNameSpecifier(const NestedNameSpecifier *NNS) {
  assert(NNS && "Expecting non-null pointer.");
  const auto *Prefix = NNS->getPrefix();
  AddBoolean(Prefix);
  if (Prefix)
    AddNestedNameSpecifier(Prefix);

  auto Kind = NNS->getKind();
  ID.AddInteger(Kind);
  switch (Kind) {
  case NestedNameSpecifier::Identifier:
    AddIdentifierInfo(NNS->getAsIdentifier());
    break;
  case NestedNameSpecifier::Namespace:
    AddDecl(NNS->getAsNamespace());
    break;
  case NestedNameSpecifier::NamespaceAlias:
    AddDecl(NNS->getAsNamespaceAlias());
    break;
  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    AddType(NNS->getAsType());
    break;
  case NestedNameSpecifier::Global:
  case NestedNameSpecifier::Super:
    break;
  }
}

void ODRHash::AddTemplateName(TemplateName Name) {
  const auto Kind = Name.getKind();
  ID.AddInteger(Kind);
  AddBoolean(Name.isDependent());
  AddBoolean(Name.isInstantiationDependent());
  switch (Kind) {
  case TemplateName::Template:
    AddDecl(Name.getAsTemplateDecl());
    break;
  case TemplateName::OverloadedTemplate: {
    const auto *Storage = Name.getAsOverloadedTemplate();
    ID.AddInteger(Storage->size());
    for (const auto *ND : *Storage) {
      AddDecl(ND);
    }
    break;
  }
  case TemplateName::QualifiedTemplate: {
    const auto *QTN = Name.getAsQualifiedTemplateName();
    AddNestedNameSpecifier(QTN->getQualifier());
    AddBoolean(QTN->hasTemplateKeyword());
    AddDecl(QTN->getDecl());
    break;
  }
  case TemplateName::DependentTemplate: {
    const auto *DTN = Name.getAsDependentTemplateName();
    AddBoolean(DTN->isIdentifier());
    if (DTN->isIdentifier()) {
      AddIdentifierInfo(DTN->getIdentifier());
    } else {
      ID.AddInteger(DTN->getOperator());
    }
    break;
  }
  case TemplateName::SubstTemplateTemplateParm: {
    const auto *Storage = Name.getAsSubstTemplateTemplateParm();
    AddDecl(Storage->getParameter());
    AddTemplateName(Storage->getReplacement());
    break;
  }
  case TemplateName::SubstTemplateTemplateParmPack: {
    const auto *Storage = Name.getAsSubstTemplateTemplateParmPack();
    AddDecl(Storage->getParameterPack());
    AddTemplateArgument(Storage->getArgumentPack());
    break;
  }
  }
}

void ODRHash::AddDeclarationName(DeclarationName Name) {
  AddBoolean(Name.isEmpty());
  if (Name.isEmpty())
    return;

  auto Kind = Name.getNameKind();
  ID.AddInteger(Kind);
  switch (Kind) {
  case DeclarationName::Identifier:
    AddIdentifierInfo(Name.getAsIdentifierInfo());
    break;
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector: {
    Selector S = Name.getObjCSelector();
    AddBoolean(S.isNull());
    AddBoolean(S.isKeywordSelector());
    AddBoolean(S.isUnarySelector());
    unsigned NumArgs = S.getNumArgs();
    for (unsigned i = 0; i < NumArgs; ++i) {
      AddIdentifierInfo(S.getIdentifierInfoForSlot(i));
    }
    break;
  }
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
    AddQualType(Name.getCXXNameType());
    break;
  case DeclarationName::CXXOperatorName:
    ID.AddInteger(Name.getCXXOverloadedOperator());
    break;
  case DeclarationName::CXXLiteralOperatorName:
    AddIdentifierInfo(Name.getCXXLiteralIdentifier());
    break;
  case DeclarationName::CXXConversionFunctionName:
    AddQualType(Name.getCXXNameType());
    break;
  case DeclarationName::CXXUsingDirective:
    break;
  case DeclarationName::CXXDeductionGuideName: {
    auto *Template = Name.getCXXDeductionGuideTemplate();
    AddBoolean(Template);
    if (Template) {
      AddDecl(Template);
    }
  }
  }
}

void ODRHash::AddTemplateArgument(TemplateArgument TA) {
  const auto Kind = TA.getKind();
  ID.AddInteger(Kind);
  switch (Kind) {
  case TemplateArgument::Null:
    llvm_unreachable("Require valid TemplateArgument");
  case TemplateArgument::Type:
    AddQualType(TA.getAsType());
    break;
  case TemplateArgument::Declaration:
    AddDecl(TA.getAsDecl());
    break;
  case TemplateArgument::NullPtr:
    AddQualType(TA.getNullPtrType());
    break;
  case TemplateArgument::Integral:
    TA.getAsIntegral().Profile(ID);
    AddQualType(TA.getIntegralType());
    break;
  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    AddTemplateName(TA.getAsTemplateOrTemplatePattern());
    break;
  case TemplateArgument::Expression:
    AddStmt(TA.getAsExpr());
    break;
  case TemplateArgument::Pack:
    ID.AddInteger(TA.pack_size());
    for (auto SubTA : TA.pack_elements())
      AddTemplateArgument(SubTA);
    break;
  }
}

void ODRHash::AddTemplateParameterList(const TemplateParameterList *TPL) {
  assert(TPL && "Expecting non-null pointer.");
  ID.AddInteger(TPL->size());
  for (auto *ND : TPL->asArray()) {
    AddSubDecl(ND);
  }
}

void ODRHash::clear() {
  DeclMap.clear();
  TypeMap.clear();
  Bools.clear();
  ID.clear();
}

unsigned ODRHash::CalculateHash() {
  // Append the bools to the end of the data segment backwards.  This allows
  // for the bools data to be compressed 32 times smaller compared to using
  // ID.AddBoolean
  const unsigned unsigned_bits = sizeof(unsigned) * CHAR_BIT;
  const unsigned size = Bools.size();
  const unsigned remainder = size % unsigned_bits;
  const unsigned loops = size / unsigned_bits;
  auto I = Bools.rbegin();
  unsigned value = 0;
  for (unsigned i = 0; i < remainder; ++i) {
    value <<= 1;
    value |= *I;
    ++I;
  }
  ID.AddInteger(value);

  for (unsigned i = 0; i < loops; ++i) {
    value = 0;
    for (unsigned j = 0; j < unsigned_bits; ++j) {
      value <<= 1;
      value |= *I;
      ++I;
    }
    ID.AddInteger(value);
  }

  assert(I == Bools.rend());
  Bools.clear();
  return ID.ComputeHash();
}

// Process a Decl pointer.  Add* methods call back into ODRHash while Visit*
// methods process the relevant parts of the Decl.
class ODRDeclVisitor : public ConstDeclVisitor<ODRDeclVisitor> {
  typedef ConstDeclVisitor<ODRDeclVisitor> Inherited;
  llvm::FoldingSetNodeID &ID;
  ODRHash &Hash;

public:
  ODRDeclVisitor(llvm::FoldingSetNodeID &ID, ODRHash &Hash)
      : ID(ID), Hash(Hash) {}

  void AddDecl(const Decl *D) {
    Hash.AddBoolean(D);
    if (D) {
      Hash.AddDecl(D);
    }
  }

  void AddStmt(const Stmt *S) {
    Hash.AddBoolean(S);
    if (S) {
      Hash.AddStmt(S);
    }
  }

  void AddQualType(QualType T) {
    Hash.AddQualType(T);
  }

  void AddIdentifierInfo(const IdentifierInfo *II) {
    Hash.AddBoolean(II);
    if (II) {
      Hash.AddIdentifierInfo(II);
    }
  }

  void AddTemplateParameterList(TemplateParameterList *TPL) {
    Hash.AddBoolean(TPL);
    if (TPL) {
      Hash.AddTemplateParameterList(TPL);
    }
  }

  void AddTemplateArgument(TemplateArgument TA) {
    Hash.AddTemplateArgument(TA);
  }

  void Visit(const Decl *D) {
    if (!D)
      return;
    if (D->isImplicit())
      return;
    if (D->isInvalidDecl())
      return;
    ID.AddInteger(D->getKind());

    Inherited::Visit(D);
  }

  void VisitDecl(const Decl *D) {
    Inherited::VisitDecl(D);
  }

  void VisitLabelDecl(const LabelDecl *D) {
    Inherited::VisitLabelDecl(D);
  }

  void VisitEnumDecl(const EnumDecl *D) {
    const bool isFixed = D->isFixed();
    Hash.AddBoolean(isFixed);
    if (isFixed)
      AddQualType(D->getIntegerType());
    Hash.AddBoolean(D->isScoped());
    Hash.AddBoolean(D->isScopedUsingClassTag());

    // TODO: Enums should have their own ODR hash.
    for (auto *SubDecl : D->decls()) {
      Hash.AddSubDecl(SubDecl);
    }

    Inherited::VisitEnumDecl(D);
  }

  void VisitEnumConstantDecl(const EnumConstantDecl *D) {
    auto *E = D->getInitExpr();
    AddStmt(E);

    Inherited::VisitEnumConstantDecl(D);
  }

  void VisitNamedDecl(const NamedDecl *D) {
    AddIdentifierInfo(D->getIdentifier());
    Inherited::VisitNamedDecl(D);
  }

  void VisitValueDecl(const ValueDecl *D) {
    AddQualType(D->getType());
    Inherited::VisitValueDecl(D);
  }

  void VisitParmVarDecl(const ParmVarDecl *D) {
    AddStmt(D->getDefaultArg());
    Inherited::VisitParmVarDecl(D);
  }

  void VisitAccessSpecDecl(const AccessSpecDecl *D) {
    ID.AddInteger(D->getAccess());
    Inherited::VisitAccessSpecDecl(D);
  }

  void VisitFriendDecl(const FriendDecl *D) {
    TypeSourceInfo *TSI = D->getFriendType();
    Hash.AddBoolean(TSI);
    if (TSI)
      AddQualType(TSI->getType());
    else
      AddDecl(D->getFriendDecl());

    unsigned NumLists = D->getFriendTypeNumTemplateParameterLists();
    ID.AddInteger(NumLists);
    for (unsigned i = 0; i < NumLists; ++i)
      AddTemplateParameterList(D->getFriendTypeTemplateParameterList(i));
    Inherited::VisitFriendDecl(D);
  }

  void VisitStaticAssertDecl(const StaticAssertDecl *D) {
    AddStmt(D->getAssertExpr());
    AddStmt(D->getMessage());

    Inherited::VisitStaticAssertDecl(D);
  }

  void VisitTypedefNameDecl(const TypedefNameDecl *D) {
    AddQualType(D->getUnderlyingType());

    Inherited::VisitTypedefNameDecl(D);
  }

  void VisitFunctionDecl(const FunctionDecl *D) {
    // TODO: Functions should have their own ODR hashes.
    AddStmt(D->hasBody() ? D->getBody() : nullptr);

    ID.AddInteger(D->getStorageClass());
    Hash.AddBoolean(D->isInlineSpecified());
    Hash.AddBoolean(D->isVirtualAsWritten());
    Hash.AddBoolean(D->isPure());
    Hash.AddBoolean(D->isDeletedAsWritten());
    ID.AddInteger(D->getOverloadedOperator());
    Inherited::VisitFunctionDecl(D);
  }

  void VisitCXXMethodDecl(const CXXMethodDecl *D) {
    Hash.AddBoolean(D->isStatic());
    Hash.AddBoolean(D->isInstance());
    Hash.AddBoolean(D->isConst());
    Hash.AddBoolean(D->isVolatile());
    Inherited::VisitCXXMethodDecl(D);
  }

  void VisitCXXConstructorDecl(const CXXConstructorDecl *D) {
    Hash.AddBoolean(D->isExplicitSpecified());
    unsigned NumCtorInits = 0;
    llvm::SmallVector<CXXCtorInitializer *, 4> Initializers;
    ID.AddInteger(D->getNumCtorInitializers());
    for (auto Initializer : D->inits()) {
      if (Initializer->isWritten()) {
        ++NumCtorInits;
        Initializers.push_back(Initializer);
      }
    }
    for (auto Initializer : Initializers) {
      AddStmt(Initializer->getInit());
    }

    Inherited::VisitCXXConstructorDecl(D);
  }

  void VisitCXXConversionDecl(const CXXConversionDecl *D) {
    AddQualType(D->getConversionType());
    Hash.AddBoolean(D->isExplicitSpecified());
    Inherited::VisitCXXConversionDecl(D);
  }

  void VisitFieldDecl(const FieldDecl *D) {
    Hash.AddBoolean(D->isMutable());

    const bool isBitField = D->isBitField();
    Hash.AddBoolean(isBitField);
    if (isBitField)
      AddStmt(D->getBitWidth());

    AddStmt(D->getInClassInitializer());

    Inherited::VisitFieldDecl(D);
  }

  void VisitTemplateDecl(const TemplateDecl *D) {
    AddDecl(D->getTemplatedDecl());

    auto *Parameters = D->getTemplateParameters();
    ID.AddInteger(Parameters->size());
    for (auto *ND : *Parameters)
      AddDecl(ND);

    Inherited::VisitTemplateDecl(D);
  }

  void VisitFunctionTemplateDecl(const FunctionTemplateDecl *D) {
    Inherited::VisitFunctionTemplateDecl(D);
  }

  void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D) {
    const bool hasDefaultArgument = D->hasDefaultArgument();
    Hash.AddBoolean(hasDefaultArgument);
    if (hasDefaultArgument)
      AddTemplateArgument(D->getDefaultArgument());

    Inherited::VisitTemplateTypeParmDecl(D);
  }

  void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D) {
    AddStmt(D->hasDefaultArgument() ? D->getDefaultArgument() : nullptr);

    Inherited::VisitNonTypeTemplateParmDecl(D);
  }

  void VisitTemplateTemplateParmDecl(const TemplateTemplateParmDecl *D) {
    const bool hasDefaultArgument = D->hasDefaultArgument();
    Hash.AddBoolean(hasDefaultArgument);
    if (hasDefaultArgument)
      AddTemplateArgument(D->getDefaultArgument().getArgument());

    Inherited::VisitTemplateTemplateParmDecl(D);
  }

  void VisitCXXRecordDecl(const CXXRecordDecl *D) {
    const bool hasDefinition = D->hasDefinition();
    Hash.AddBoolean(hasDefinition);
    if (hasDefinition) {
      ID.AddInteger(D->getODRHash());
    }
    Inherited::VisitCXXRecordDecl(D);
  }
};


void ODRHash::AddSubDecl(const Decl *D) {
  assert(D && "Expecting non-null pointer.");
  AddDecl(D);
  ODRDeclVisitor(ID, *this).Visit(D);
}

void ODRHash::AddCXXRecordDecl(const CXXRecordDecl *Record) {
  assert(Record && Record->hasDefinition() &&
         "Expected non-null record to be a definition.");
  AddDecl(Record);

  // Filter out sub-Decls which will not be processed in order to get an
  // accurate count of Decl's.
  llvm::SmallVector<const Decl *, 16> Decls;
  for (const Decl *SubDecl : Record->decls()) {
    // Ignore implicit Decl's.
    if (SubDecl->isImplicit()) {
      continue;
    }
    // Ignore Decl's that are not in the context of the CXXRecordDecl.
    if (SubDecl->getDeclContext() != Record) {
      continue;
    }
    Decls.push_back(SubDecl);
  }

  ID.AddInteger(Decls.size());
  for (auto SubDecl : Decls) {
    AddSubDecl(SubDecl);
  }

  ID.AddInteger(Record->getNumBases());
  for (auto base : Record->bases()) {
    AddBoolean(base.isVirtual());
    AddQualType(base.getType());
  }

  const ClassTemplateDecl *TD = Record->getDescribedClassTemplate();
  AddBoolean(TD);
  if (TD) {
    AddTemplateParameterList(TD->getTemplateParameters());
  }
}

void ODRHash::AddDecl(const Decl *D) {
  assert(D && "Expecting non-null pointer.");
  auto Result = DeclMap.insert(std::make_pair(D, DeclMap.size()));
  ID.AddInteger(Result.first->second);
  // On first encounter of a Decl pointer, process it.  Every time afterwards,
  // only the index value is needed.
  if (!Result.second) {
    return;
  }

  // Unlike the DeclVisitor, this adds a limited amount of information to
  // identify the Decl.
  ID.AddInteger(D->getKind());

  // Unlike other places where AddBoolean is used with possibly null pointers,
  // the nullness of the following pointers is already encoded with the
  // DeclKind value, so there is no ambiguity on what information is added.
  if (const auto *ND = dyn_cast<NamedDecl>(D)) {
    AddDeclarationName(ND->getDeclName());
  }

  if (const auto *Typedef = dyn_cast<TypedefNameDecl>(D)) {
    AddQualType(Typedef->getUnderlyingType());
  }

  if (const auto *Alias = dyn_cast<NamespaceAliasDecl>(D)) {
    AddDecl(Alias->getNamespace());
  }
}

// Process a Type pointer.  Add* methods call back into ODRHash while Visit*
// methods process the relevant parts of the Type.
class ODRTypeVisitor : public TypeVisitor<ODRTypeVisitor> {
  typedef TypeVisitor<ODRTypeVisitor> Inherited;
  llvm::FoldingSetNodeID &ID;
  ODRHash &Hash;

public:
  ODRTypeVisitor(llvm::FoldingSetNodeID &ID, ODRHash &Hash)
      : ID(ID), Hash(Hash) {}

  void AddType(const Type *T) {
    Hash.AddType(T);
  }

  void AddQualType(QualType T) {
    Hash.AddQualType(T);
  }

  void AddDecl(Decl *D) {
    Hash.AddBoolean(D);
    if (D) {
      Hash.AddDecl(D);
    }
  }

  void AddTemplateArgument(TemplateArgument TA) {
    Hash.AddTemplateArgument(TA);
  }

  void AddStmt(Stmt *S) {
    Hash.AddBoolean(S);
    if (S) {
      Hash.AddStmt(S);
    }
  }

  void AddNestedNameSpecifier(NestedNameSpecifier *NNS) {
    Hash.AddBoolean(NNS);
    if (NNS) {
      Hash.AddNestedNameSpecifier(NNS);
    }
  }
  void AddIdentiferInfo(const IdentifierInfo *II) {
    Hash.AddBoolean(II);
    if (II) {
      Hash.AddIdentifierInfo(II);
    }
  }

  void AddTemplateName(TemplateName TN) {
    Hash.AddTemplateName(TN);
  }

  void VisitQualifiers(Qualifiers Quals) {
    ID.AddInteger(Quals.getAsOpaqueValue());
  }

  void Visit(const Type *T) {
    ID.AddInteger(T->getTypeClass());
    Inherited::Visit(T);
  }

  void VisitType(const Type *T) {}

  void VisitAdjustedType(const AdjustedType *T) {
    AddQualType(T->getOriginalType());
    AddQualType(T->getAdjustedType());
    VisitType(T);
  }

  void VisitDecayedType(const DecayedType *T) {
    AddQualType(T->getDecayedType());
    AddQualType(T->getPointeeType());
    VisitAdjustedType(T);
  }

  void VisitArrayType(const ArrayType *T) {
    AddQualType(T->getElementType());
    ID.AddInteger(T->getSizeModifier());
    VisitQualifiers(T->getIndexTypeQualifiers());
    VisitType(T);
  }
  void VisitConstantArrayType(const ConstantArrayType *T) {
    T->getSize().Profile(ID);
    VisitArrayType(T);
  }

  void VisitDependentSizedArrayType(const DependentSizedArrayType *T) {
    AddStmt(T->getSizeExpr());
    VisitArrayType(T);
  }

  void VisitIncompleteArrayType(const IncompleteArrayType *T) {
    VisitArrayType(T);
  }

  void VisitVariableArrayType(const VariableArrayType *T) {
    AddStmt(T->getSizeExpr());
    VisitArrayType(T);
  }

  void VisitAtomicType(const AtomicType *T) {
    AddQualType(T->getValueType());
    VisitType(T);
  }

  void VisitAttributedType(const AttributedType *T) {
    ID.AddInteger(T->getAttrKind());
    AddQualType(T->getModifiedType());
    AddQualType(T->getEquivalentType());
    VisitType(T);
  }

  void VisitBlockPointerType(const BlockPointerType *T) {
    AddQualType(T->getPointeeType());
    VisitType(T);
  }

  void VisitBuiltinType(const BuiltinType *T) {
    ID.AddInteger(T->getKind());
    VisitType(T);
  }

  void VisitComplexType(const ComplexType *T) {
    AddQualType(T->getElementType());
    VisitType(T);
  }

  void VisitDecltypeType(const DecltypeType *T) {
    AddQualType(T->getUnderlyingType());
    AddStmt(T->getUnderlyingExpr());
    VisitType(T);
  }

  void VisitDependentSizedExtVectorType(const DependentSizedExtVectorType *T) {
    AddQualType(T->getElementType());
    AddStmt(T->getSizeExpr());
    VisitType(T);
  }

  void VisitFunctionType(const FunctionType *T) {
    AddQualType(T->getReturnType());
    T->getExtInfo().Profile(ID);
    Hash.AddBoolean(T->isConst());
    Hash.AddBoolean(T->isVolatile());
    Hash.AddBoolean(T->isRestrict());
    VisitType(T);
  }

  void VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
    VisitFunctionType(T);
  }

  void VisitFunctionProtoType(const FunctionProtoType *T) {
    ID.AddInteger(T->getNumParams());
    for (auto ParamType : T->getParamTypes())
      AddQualType(ParamType);

    const auto &epi = T->getExtProtoInfo();
    ID.AddInteger(epi.Variadic);
    ID.AddInteger(epi.TypeQuals);
    ID.AddInteger(epi.RefQualifier);
    ID.AddInteger(epi.ExceptionSpec.Type);

    if (epi.ExceptionSpec.Type == EST_Dynamic) {
      for (QualType Ex : epi.ExceptionSpec.Exceptions)
        AddQualType(Ex);
    } else if (epi.ExceptionSpec.Type == EST_ComputedNoexcept &&
               epi.ExceptionSpec.NoexceptExpr) {
      AddStmt(epi.ExceptionSpec.NoexceptExpr);
    } else if (epi.ExceptionSpec.Type == EST_Uninstantiated ||
               epi.ExceptionSpec.Type == EST_Unevaluated) {
      AddDecl(epi.ExceptionSpec.SourceDecl->getCanonicalDecl());
    }
    if (epi.ExtParameterInfos) {
      for (unsigned i = 0; i != T->getNumParams(); ++i)
        ID.AddInteger(epi.ExtParameterInfos[i].getOpaqueValue());
    }
    epi.ExtInfo.Profile(ID);
    Hash.AddBoolean(epi.HasTrailingReturn);

    VisitFunctionType(T);
  }

  void VisitInjectedClassNameType(const InjectedClassNameType *T) {
    AddDecl(T->getDecl());
    VisitType(T);
  }

  void VisitMemberPointerType(const MemberPointerType *T) {
    AddQualType(T->getPointeeType());
    AddType(T->getClass());
    VisitType(T);
  }

  void VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
    AddQualType(T->getPointeeType());
    VisitType(T);
  }

  void VisitObjCObjectType(const ObjCObjectType *T) {
    QualType Base = T->getBaseType();
    Hash.AddBoolean(Base.getTypePtr() != T);
    if (Base.getTypePtr() != T)
      AddQualType(Base);
    auto TypeArgs = T->getTypeArgsAsWritten();
    ID.AddInteger(TypeArgs.size());
    for (auto TypeArg : TypeArgs)
      AddQualType(TypeArg);
    ID.AddInteger(T->getNumProtocols());
    for (auto proto : T->quals())
      AddDecl(proto);
    ID.AddInteger(T->isKindOfTypeAsWritten());
    VisitType(T);
  }

  void VisitObjCInterfaceType(const ObjCInterfaceType *T) {
    VisitObjCObjectType(T);
  }

  void VisitObjCObjectTypeImpl(const ObjCObjectTypeImpl *T) {
    VisitObjCObjectType(T);
  }

  void VisitPackExpansionType(const PackExpansionType *T) {
    AddQualType(T->getPattern());
    auto NumExpansions = T->getNumExpansions();
    Hash.AddBoolean(NumExpansions.hasValue());
    if (NumExpansions)
      ID.AddInteger(*NumExpansions);
    VisitType(T);
  };

  void VisitPointerType(const PointerType *T) {
    AddQualType(T->getPointeeType());
    VisitType(T);
  }

  void VisitReferenceType(const ReferenceType *T) {
    AddQualType(T->getPointeeTypeAsWritten());
    VisitType(T);
  }

  void VisitLValueReferenceType(const LValueReferenceType *T) {
    VisitReferenceType(T);
  }

  void VisitRValueReferenceType(const RValueReferenceType *T) {
    VisitReferenceType(T);
  }

  void VisitSubstTemplateTypeParmType(const SubstTemplateTypeParmType *T) {
    AddQualType(T->getReplacementType());
    AddType(T->getReplacedParameter());
    VisitType(T);
  }

  void
  VisitSubstTemplateTypeParmPackType(const SubstTemplateTypeParmPackType *T) {
    AddType(T->getReplacedParameter());
    AddTemplateArgument(T->getArgumentPack());
    VisitType(T);
  }

  void VisitTagType(const TagType *T) {
    AddDecl(T->getDecl());
    Hash.AddBoolean(T->isBeingDefined());
    VisitType(T);
  }

  void VisitEnumType(const EnumType *T) {
    AddDecl(T->getDecl());
    VisitTagType(T);
  }

  void VisitRecordType(const RecordType *T) {
    AddDecl(T->getDecl());
    VisitTagType(T);
  }

  void VisitTemplateSpecializationType(const TemplateSpecializationType *T) {
    AddTemplateName(T->getTemplateName());
    ID.AddInteger(T->getNumArgs());
    for (auto I = T->begin(), E = T->end(); I != E; ++I)
      AddTemplateArgument(*I);
    VisitType(T);
  }

  void VisitTemplateTypeParmType(const TemplateTypeParmType *T) {
    ID.AddInteger(T->getDepth());
    ID.AddInteger(T->getIndex());
    Hash.AddBoolean(T->isParameterPack());
    AddDecl(T->getDecl());
    VisitType(T);
  }

  void VisitTypedefType(const TypedefType *T) {
    AddDecl(T->getDecl());
    VisitType(T);
  }

  void VisitTypeOfExprType(const TypeOfExprType *T) {
    AddStmt(T->getUnderlyingExpr());
    VisitType(T);
  }

  void VisitDependentTypeOfExprType(const DependentTypeOfExprType *T) {
    VisitTypeOfExprType(T);
  }

  void VisitTypeWithKeyword(const TypeWithKeyword *T) { VisitType(T); }

  void VisitElaboratedType(const ElaboratedType *T) {
    ID.AddInteger(T->getKeyword());
    AddNestedNameSpecifier(T->getQualifier());
    AddQualType(T->getNamedType());
    VisitTypeWithKeyword(T);
  }

  void VisitUnaryTransformType(const UnaryTransformType *T) {
    AddQualType(T->getBaseType());
    ID.AddInteger(T->getUTTKind());
    VisitType(T);
  }

  void VisitDependentUnaryTransformType(const DependentUnaryTransformType *T) {
    VisitUnaryTransformType(T);
  }

  void VisitUnresolvedUsingType(const UnresolvedUsingType *T) {
    AddDecl(T->getDecl());
    VisitType(T);
  }

  void VisitVectorType(const VectorType *T) {
    AddQualType(T->getElementType());
    ID.AddInteger(T->getNumElements());
    ID.AddInteger(T->getVectorKind());
    VisitType(T);
  }

  void VisitExtVectorType(const ExtVectorType *T) { VisitVectorType(T); }
};

void ODRHash::AddType(const Type *T) {
  assert(T && "Expecting non-null pointer.");
  auto Result = TypeMap.insert(std::make_pair(T, TypeMap.size()));
  ID.AddInteger(Result.first->second);
  // On first encounter of a Type pointer, process it.  Every time afterwards,
  // only the index value is needed.
  if (!Result.second) {
    return;
  }

  ODRTypeVisitor(ID, *this).Visit(T);
}

void ODRHash::AddQualType(QualType T) {
  AddBoolean(T.isNull());
  if (T.isNull())
    return;
  SplitQualType split = T.split();
  ID.AddInteger(split.Quals.getAsOpaqueValue());
  AddType(split.Ty);
}

void ODRHash::AddBoolean(bool Value) {
  Bools.push_back(Value);
}
