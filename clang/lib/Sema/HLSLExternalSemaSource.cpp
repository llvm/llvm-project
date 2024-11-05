//===--- HLSLExternalSemaSource.cpp - HLSL Sema Source --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/HLSLExternalSemaSource.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/HLSLRuntime.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"

#include <functional>

using namespace clang;
using namespace llvm::hlsl;

namespace {

struct TemplateParameterListBuilder;

struct BuiltinTypeDeclBuilder {
  CXXRecordDecl *Record = nullptr;
  ClassTemplateDecl *Template = nullptr;
  ClassTemplateDecl *PrevTemplate = nullptr;
  NamespaceDecl *HLSLNamespace = nullptr;
  llvm::StringMap<FieldDecl *> Fields;

  BuiltinTypeDeclBuilder(CXXRecordDecl *R) : Record(R) {
    Record->startDefinition();
    Template = Record->getDescribedClassTemplate();
  }

  BuiltinTypeDeclBuilder(Sema &S, NamespaceDecl *Namespace, StringRef Name)
      : HLSLNamespace(Namespace) {
    ASTContext &AST = S.getASTContext();
    IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);

    LookupResult Result(S, &II, SourceLocation(), Sema::LookupTagName);
    CXXRecordDecl *PrevDecl = nullptr;
    if (S.LookupQualifiedName(Result, HLSLNamespace)) {
      NamedDecl *Found = Result.getFoundDecl();
      if (auto *TD = dyn_cast<ClassTemplateDecl>(Found)) {
        PrevDecl = TD->getTemplatedDecl();
        PrevTemplate = TD;
      } else
        PrevDecl = dyn_cast<CXXRecordDecl>(Found);
      assert(PrevDecl && "Unexpected lookup result type.");
    }

    if (PrevDecl && PrevDecl->isCompleteDefinition()) {
      Record = PrevDecl;
      return;
    }

    Record = CXXRecordDecl::Create(AST, TagDecl::TagKind::Class, HLSLNamespace,
                                   SourceLocation(), SourceLocation(), &II,
                                   PrevDecl, true);
    Record->setImplicit(true);
    Record->setLexicalDeclContext(HLSLNamespace);
    Record->setHasExternalLexicalStorage();

    // Don't let anyone derive from built-in types.
    Record->addAttr(FinalAttr::CreateImplicit(AST, SourceRange(),
                                              FinalAttr::Keyword_final));
  }

  ~BuiltinTypeDeclBuilder() {
    if (HLSLNamespace && !Template && Record->getDeclContext() == HLSLNamespace)
      HLSLNamespace->addDecl(Record);
  }

  BuiltinTypeDeclBuilder &
  addMemberVariable(StringRef Name, QualType Type, llvm::ArrayRef<Attr *> Attrs,
                    AccessSpecifier Access = AccessSpecifier::AS_private) {
    if (Record->isCompleteDefinition())
      return *this;
    assert(Record->isBeingDefined() &&
           "Definition must be started before adding members!");
    ASTContext &AST = Record->getASTContext();

    IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);
    TypeSourceInfo *MemTySource =
        AST.getTrivialTypeSourceInfo(Type, SourceLocation());
    auto *Field = FieldDecl::Create(
        AST, Record, SourceLocation(), SourceLocation(), &II, Type, MemTySource,
        nullptr, false, InClassInitStyle::ICIS_NoInit);
    Field->setAccess(Access);
    Field->setImplicit(true);
    for (Attr *A : Attrs) {
      if (A)
        Field->addAttr(A);
    }

    Record->addDecl(Field);
    Fields[Name] = Field;
    return *this;
  }

  BuiltinTypeDeclBuilder &
  addHandleMember(Sema &S, ResourceClass RC, ResourceKind RK, bool IsROV,
                  bool RawBuffer,
                  AccessSpecifier Access = AccessSpecifier::AS_private) {
    if (Record->isCompleteDefinition())
      return *this;

    ASTContext &Ctx = S.getASTContext();
    TypeSourceInfo *ElementTypeInfo = nullptr;

    QualType ElemTy = Ctx.Char8Ty;
    if (Template) {
      if (const auto *TTD = dyn_cast<TemplateTypeParmDecl>(
              Template->getTemplateParameters()->getParam(0))) {
        ElemTy = QualType(TTD->getTypeForDecl(), 0);
      }
    }
    ElementTypeInfo = Ctx.getTrivialTypeSourceInfo(ElemTy, SourceLocation());

    // add handle member with resource type attributes
    QualType AttributedResTy = QualType();
    SmallVector<const Attr *> Attrs = {
        HLSLResourceClassAttr::CreateImplicit(Ctx, RC),
        IsROV ? HLSLROVAttr::CreateImplicit(Ctx) : nullptr,
        RawBuffer ? HLSLRawBufferAttr::CreateImplicit(Ctx) : nullptr,
        ElementTypeInfo
            ? HLSLContainedTypeAttr::CreateImplicit(Ctx, ElementTypeInfo)
            : nullptr};
    Attr *ResourceAttr = HLSLResourceAttr::CreateImplicit(Ctx, RK);
    if (CreateHLSLAttributedResourceType(S, Ctx.HLSLResourceTy, Attrs,
                                         AttributedResTy))
      addMemberVariable("h", AttributedResTy, {ResourceAttr}, Access);
    return *this;
  }

  static DeclRefExpr *lookupBuiltinFunction(ASTContext &AST, Sema &S,
                                            StringRef Name) {
    IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);
    DeclarationNameInfo NameInfo =
        DeclarationNameInfo(DeclarationName(&II), SourceLocation());
    LookupResult R(S, NameInfo, Sema::LookupOrdinaryName);
    // AllowBuiltinCreation is false but LookupDirect will create
    // the builtin when searching the global scope anyways...
    S.LookupName(R, S.getCurScope());
    // FIXME: If the builtin function was user-declared in global scope,
    // this assert *will* fail. Should this call LookupBuiltin instead?
    assert(R.isSingleResult() &&
           "Since this is a builtin it should always resolve!");
    auto *VD = cast<ValueDecl>(R.getFoundDecl());
    QualType Ty = VD->getType();
    return DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), SourceLocation(),
                               VD, false, NameInfo, Ty, VK_PRValue);
  }

  static Expr *emitResourceClassExpr(ASTContext &AST, ResourceClass RC) {
    return IntegerLiteral::Create(
        AST,
        llvm::APInt(AST.getIntWidth(AST.UnsignedCharTy),
                    static_cast<uint8_t>(RC)),
        AST.UnsignedCharTy, SourceLocation());
  }

  BuiltinTypeDeclBuilder &addDefaultHandleConstructor(Sema &S,
                                                      ResourceClass RC) {
    if (Record->isCompleteDefinition())
      return *this;
    ASTContext &AST = Record->getASTContext();

    QualType ConstructorType =
        AST.getFunctionType(AST.VoidTy, {}, FunctionProtoType::ExtProtoInfo());

    CanQualType CanTy = Record->getTypeForDecl()->getCanonicalTypeUnqualified();
    DeclarationName Name = AST.DeclarationNames.getCXXConstructorName(CanTy);
    CXXConstructorDecl *Constructor = CXXConstructorDecl::Create(
        AST, Record, SourceLocation(),
        DeclarationNameInfo(Name, SourceLocation()), ConstructorType,
        AST.getTrivialTypeSourceInfo(ConstructorType, SourceLocation()),
        ExplicitSpecifier(), false, true, false,
        ConstexprSpecKind::Unspecified);

    Constructor->setBody(CompoundStmt::Create(
        AST, {}, FPOptionsOverride(), SourceLocation(), SourceLocation()));
    Constructor->setAccess(AccessSpecifier::AS_public);
    Record->addDecl(Constructor);
    return *this;
  }

  BuiltinTypeDeclBuilder &addArraySubscriptOperators() {
    if (Record->isCompleteDefinition())
      return *this;
    addArraySubscriptOperator(true);
    addArraySubscriptOperator(false);
    return *this;
  }

  BuiltinTypeDeclBuilder &addArraySubscriptOperator(bool IsConst) {
    if (Record->isCompleteDefinition())
      return *this;

    ASTContext &AST = Record->getASTContext();
    QualType ElemTy = AST.Char8Ty;
    if (Template) {
      if (const auto *TTD = dyn_cast<TemplateTypeParmDecl>(
              Template->getTemplateParameters()->getParam(0))) {
        ElemTy = QualType(TTD->getTypeForDecl(), 0);
      }
    }
    QualType ReturnTy = ElemTy;

    FunctionProtoType::ExtProtoInfo ExtInfo;

    // Subscript operators return references to elements, const makes the
    // reference and method const so that the underlying data is not mutable.
    ReturnTy = AST.getLValueReferenceType(ReturnTy);
    if (IsConst) {
      ExtInfo.TypeQuals.addConst();
      ReturnTy.addConst();
    }

    QualType MethodTy =
        AST.getFunctionType(ReturnTy, {AST.UnsignedIntTy}, ExtInfo);
    auto *TSInfo = AST.getTrivialTypeSourceInfo(MethodTy, SourceLocation());
    auto *MethodDecl = CXXMethodDecl::Create(
        AST, Record, SourceLocation(),
        DeclarationNameInfo(
            AST.DeclarationNames.getCXXOperatorName(OO_Subscript),
            SourceLocation()),
        MethodTy, TSInfo, SC_None, false, false, ConstexprSpecKind::Unspecified,
        SourceLocation());

    IdentifierInfo &II = AST.Idents.get("Idx", tok::TokenKind::identifier);
    auto *IdxParam = ParmVarDecl::Create(
        AST, MethodDecl->getDeclContext(), SourceLocation(), SourceLocation(),
        &II, AST.UnsignedIntTy,
        AST.getTrivialTypeSourceInfo(AST.UnsignedIntTy, SourceLocation()),
        SC_None, nullptr);
    MethodDecl->setParams({IdxParam});

    // Also add the parameter to the function prototype.
    auto FnProtoLoc = TSInfo->getTypeLoc().getAs<FunctionProtoTypeLoc>();
    FnProtoLoc.setParam(0, IdxParam);

    // FIXME: Placeholder to make sure we return the correct type - create
    // field of element_type and return reference to it. This field will go
    // away once indexing into resources is properly implemented in
    // llvm/llvm-project#95956.
    if (Fields.count("e") == 0) {
      addMemberVariable("e", ElemTy, {});
    }
    FieldDecl *ElemFieldDecl = Fields["e"];

    auto *This =
        CXXThisExpr::Create(AST, SourceLocation(),
                            MethodDecl->getFunctionObjectParameterType(), true);
    Expr *ElemField = MemberExpr::CreateImplicit(
        AST, This, false, ElemFieldDecl, ElemFieldDecl->getType(), VK_LValue,
        OK_Ordinary);
    auto *Return =
        ReturnStmt::Create(AST, SourceLocation(), ElemField, nullptr);

    MethodDecl->setBody(CompoundStmt::Create(AST, {Return}, FPOptionsOverride(),
                                             SourceLocation(),
                                             SourceLocation()));
    MethodDecl->setLexicalDeclContext(Record);
    MethodDecl->setAccess(AccessSpecifier::AS_public);
    MethodDecl->addAttr(AlwaysInlineAttr::CreateImplicit(
        AST, SourceRange(), AlwaysInlineAttr::CXX11_clang_always_inline));
    Record->addDecl(MethodDecl);

    return *this;
  }

  BuiltinTypeDeclBuilder &startDefinition() {
    if (Record->isCompleteDefinition())
      return *this;
    Record->startDefinition();
    return *this;
  }

  BuiltinTypeDeclBuilder &completeDefinition() {
    if (Record->isCompleteDefinition())
      return *this;
    assert(Record->isBeingDefined() &&
           "Definition must be started before completing it.");

    Record->completeDefinition();
    return *this;
  }

  TemplateParameterListBuilder addTemplateArgumentList(Sema &S);
  BuiltinTypeDeclBuilder &
  addSimpleTemplateParams(Sema &S, ArrayRef<StringRef> Names, ConceptDecl *CD);
  BuiltinTypeDeclBuilder &addConceptSpecializationExpr(Sema &S);
};

struct TemplateParameterListBuilder {
  BuiltinTypeDeclBuilder &Builder;
  Sema &S;
  llvm::SmallVector<NamedDecl *> Params;

  TemplateParameterListBuilder(Sema &S, BuiltinTypeDeclBuilder &RB)
      : Builder(RB), S(S) {}

  ~TemplateParameterListBuilder() { finalizeTemplateArgs(); }

  TemplateParameterListBuilder &
  addTypeParameter(StringRef Name, QualType DefaultValue = QualType()) {
    if (Builder.Record->isCompleteDefinition())
      return *this;
    unsigned Position = static_cast<unsigned>(Params.size());
    auto *Decl = TemplateTypeParmDecl::Create(
        S.Context, Builder.Record->getDeclContext(), SourceLocation(),
        SourceLocation(), /* TemplateDepth */ 0, Position,
        &S.Context.Idents.get(Name, tok::TokenKind::identifier),
        /* Typename */ true,
        /* ParameterPack */ false,
        /* HasTypeConstraint*/ false);
    if (!DefaultValue.isNull())
      Decl->setDefaultArgument(
          S.Context, S.getTrivialTemplateArgumentLoc(DefaultValue, QualType(),
                                                     SourceLocation()));
    Params.emplace_back(Decl);
    return *this;
  }

  /*
  The concept specialization expression (CSE) constructed below is constructed
  so that it matches the CSE that is constructed when parsing
  the below C++ code:

  template<typename T>
  concept is_valid_line_vector =sizeof(T) <= 16;

  template<typename element_type> requires is_valid_line_vector<element_type>

  struct RWBuffer {
      element_type Val;
  };

  int fn() {
      RWBuffer<int> Buf;
  }

  When dumping the AST and filtering for "RWBuffer", the resulting AST
  structure is what we're trying to construct below, specifically the
  CSE portion.
  */
  ConceptSpecializationExpr *
  constructConceptSpecializationExpr(Sema &S, ConceptDecl *CD) {
    ASTContext &Context = S.getASTContext();
    SourceLocation Loc = Builder.Record->getBeginLoc();
    DeclarationNameInfo DNI(CD->getDeclName(), Loc);
    NestedNameSpecifierLoc NNSLoc;
    DeclContext *DC = Builder.Record->getDeclContext();
    TemplateArgumentListInfo TALI(Loc, Loc);

    // Assume that the concept decl has just one template parameter
    // This parameter should have been added when CD was constructed
    // in getTypedBufferConceptDecl
    assert(CD->getTemplateParameters()->size() == 1 &&
           "unexpected concept decl parameter count");
    TemplateTypeParmDecl *ConceptTTPD = dyn_cast<TemplateTypeParmDecl>(
        CD->getTemplateParameters()->getParam(0));

    // this fake TemplateTypeParmDecl is used to construct a template argument
    // that will be used to construct the ImplicitConceptSpecializationDecl
    TemplateTypeParmDecl *T = TemplateTypeParmDecl::Create(
        Context,                          // AST context
        Context.getTranslationUnitDecl(), // DeclContext
        SourceLocation(), SourceLocation(),
        /*depth=*/0,                // Depth in the template parameter list
        /*position=*/0,             // Position in the template parameter list
        /*id=*/nullptr,             // Identifier for 'T'
        /*Typename=*/true,          // Indicates this is a 'typename' or 'class'
        /*ParameterPack=*/false,    // Not a parameter pack
        /*HasTypeConstraint=*/false // Has no type constraint
    );

    T->setDeclContext(DC);
    T->setReferenced();

    QualType ConceptTType = Context.getTypeDeclType(ConceptTTPD);

    // this is the 2nd template argument node in the AST above
    TemplateArgument ConceptTA = TemplateArgument(ConceptTType);

    QualType CSETType = Context.getTypeDeclType(T);

    // this is the 1st template argument node in the AST above
    TemplateArgument CSETA = TemplateArgument(CSETType);

    ImplicitConceptSpecializationDecl *ImplicitCSEDecl =
        ImplicitConceptSpecializationDecl::Create(
            Context, Builder.Record->getDeclContext(), Loc, {CSETA});

    // Constraint satisfaction is used to construct the
    // ConceptSpecailizationExpr, and represents the 2nd Template Argument,
    // located at the bottom of the sample AST above.
    const ConstraintSatisfaction CS(CD, {ConceptTA});
    TemplateArgumentLoc TAL = S.getTrivialTemplateArgumentLoc(
        ConceptTA, QualType(), SourceLocation());

    TALI.addArgument(TAL);
    const ASTTemplateArgumentListInfo *ATALI =
        ASTTemplateArgumentListInfo::Create(Context, TALI);

    // In the concept reference, ATALI is what adds the extra
    // TemplateArgument node underneath CSE
    ConceptReference *CR =
        ConceptReference::Create(Context, NNSLoc, Loc, DNI, CD, CD, ATALI);

    ConceptSpecializationExpr *CSE =
        ConceptSpecializationExpr::Create(Context, CR, ImplicitCSEDecl, &CS);

    return CSE;
  }

  BuiltinTypeDeclBuilder &finalizeTemplateArgs(ConceptDecl *CD = nullptr) {
    if (Params.empty())
      return Builder;
    ConceptSpecializationExpr *CSE =
        CD ? constructConceptSpecializationExpr(S, CD) : nullptr;

    auto *ParamList = TemplateParameterList::Create(S.Context, SourceLocation(),
                                                    SourceLocation(), Params,
                                                    SourceLocation(), CSE);
    Builder.Template = ClassTemplateDecl::Create(
        S.Context, Builder.Record->getDeclContext(), SourceLocation(),
        DeclarationName(Builder.Record->getIdentifier()), ParamList,
        Builder.Record);

    Builder.Record->setDescribedClassTemplate(Builder.Template);
    Builder.Template->setImplicit(true);
    Builder.Template->setLexicalDeclContext(Builder.Record->getDeclContext());

    // NOTE: setPreviousDecl before addDecl so new decl replace old decl when
    // make visible.
    Builder.Template->setPreviousDecl(Builder.PrevTemplate);
    Builder.Record->getDeclContext()->addDecl(Builder.Template);
    Params.clear();

    QualType T = Builder.Template->getInjectedClassNameSpecialization();
    T = S.Context.getInjectedClassNameType(Builder.Record, T);

    return Builder;
  }
};
} // namespace

TemplateParameterListBuilder
BuiltinTypeDeclBuilder::addTemplateArgumentList(Sema &S) {
  return TemplateParameterListBuilder(S, *this);
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addSimpleTemplateParams(
    Sema &S, ArrayRef<StringRef> Names, ConceptDecl *CD = nullptr) {
  TemplateParameterListBuilder Builder = this->addTemplateArgumentList(S);
  for (StringRef Name : Names)
    Builder.addTypeParameter(Name);

  return Builder.finalizeTemplateArgs(CD);
}

HLSLExternalSemaSource::~HLSLExternalSemaSource() {}

void HLSLExternalSemaSource::InitializeSema(Sema &S) {
  SemaPtr = &S;
  ASTContext &AST = SemaPtr->getASTContext();
  // If the translation unit has external storage force external decls to load.
  if (AST.getTranslationUnitDecl()->hasExternalLexicalStorage())
    (void)AST.getTranslationUnitDecl()->decls_begin();

  IdentifierInfo &HLSL = AST.Idents.get("hlsl", tok::TokenKind::identifier);
  LookupResult Result(S, &HLSL, SourceLocation(), Sema::LookupNamespaceName);
  NamespaceDecl *PrevDecl = nullptr;
  if (S.LookupQualifiedName(Result, AST.getTranslationUnitDecl()))
    PrevDecl = Result.getAsSingle<NamespaceDecl>();
  HLSLNamespace = NamespaceDecl::Create(
      AST, AST.getTranslationUnitDecl(), /*Inline=*/false, SourceLocation(),
      SourceLocation(), &HLSL, PrevDecl, /*Nested=*/false);
  HLSLNamespace->setImplicit(true);
  HLSLNamespace->setHasExternalLexicalStorage();
  AST.getTranslationUnitDecl()->addDecl(HLSLNamespace);

  // Force external decls in the HLSL namespace to load from the PCH.
  (void)HLSLNamespace->getCanonicalDecl()->decls_begin();
  defineTrivialHLSLTypes();
  defineHLSLTypesWithForwardDeclarations();

  // This adds a `using namespace hlsl` directive. In DXC, we don't put HLSL's
  // built in types inside a namespace, but we are planning to change that in
  // the near future. In order to be source compatible older versions of HLSL
  // will need to implicitly use the hlsl namespace. For now in clang everything
  // will get added to the namespace, and we can remove the using directive for
  // future language versions to match HLSL's evolution.
  auto *UsingDecl = UsingDirectiveDecl::Create(
      AST, AST.getTranslationUnitDecl(), SourceLocation(), SourceLocation(),
      NestedNameSpecifierLoc(), SourceLocation(), HLSLNamespace,
      AST.getTranslationUnitDecl());

  AST.getTranslationUnitDecl()->addDecl(UsingDecl);
}

void HLSLExternalSemaSource::defineHLSLVectorAlias() {
  ASTContext &AST = SemaPtr->getASTContext();

  llvm::SmallVector<NamedDecl *> TemplateParams;

  auto *TypeParam = TemplateTypeParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 0,
      &AST.Idents.get("element", tok::TokenKind::identifier), false, false);
  TypeParam->setDefaultArgument(
      AST, SemaPtr->getTrivialTemplateArgumentLoc(
               TemplateArgument(AST.FloatTy), QualType(), SourceLocation()));

  TemplateParams.emplace_back(TypeParam);

  auto *SizeParam = NonTypeTemplateParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 1,
      &AST.Idents.get("element_count", tok::TokenKind::identifier), AST.IntTy,
      false, AST.getTrivialTypeSourceInfo(AST.IntTy));
  llvm::APInt Val(AST.getIntWidth(AST.IntTy), 4);
  TemplateArgument Default(AST, llvm::APSInt(std::move(Val)), AST.IntTy,
                           /*IsDefaulted=*/true);
  SizeParam->setDefaultArgument(
      AST, SemaPtr->getTrivialTemplateArgumentLoc(Default, AST.IntTy,
                                                  SourceLocation(), SizeParam));
  TemplateParams.emplace_back(SizeParam);

  auto *ParamList =
      TemplateParameterList::Create(AST, SourceLocation(), SourceLocation(),
                                    TemplateParams, SourceLocation(), nullptr);

  IdentifierInfo &II = AST.Idents.get("vector", tok::TokenKind::identifier);

  QualType AliasType = AST.getDependentSizedExtVectorType(
      AST.getTemplateTypeParmType(0, 0, false, TypeParam),
      DeclRefExpr::Create(
          AST, NestedNameSpecifierLoc(), SourceLocation(), SizeParam, false,
          DeclarationNameInfo(SizeParam->getDeclName(), SourceLocation()),
          AST.IntTy, VK_LValue),
      SourceLocation());

  auto *Record = TypeAliasDecl::Create(AST, HLSLNamespace, SourceLocation(),
                                       SourceLocation(), &II,
                                       AST.getTrivialTypeSourceInfo(AliasType));
  Record->setImplicit(true);

  auto *Template =
      TypeAliasTemplateDecl::Create(AST, HLSLNamespace, SourceLocation(),
                                    Record->getIdentifier(), ParamList, Record);

  Record->setDescribedAliasTemplate(Template);
  Template->setImplicit(true);
  Template->setLexicalDeclContext(Record->getDeclContext());
  HLSLNamespace->addDecl(Template);
}

void HLSLExternalSemaSource::defineTrivialHLSLTypes() {
  defineHLSLVectorAlias();
}

/// Set up common members and attributes for buffer types
static BuiltinTypeDeclBuilder setupBufferType(CXXRecordDecl *Decl, Sema &S,
                                              ResourceClass RC, ResourceKind RK,
                                              bool IsROV, bool RawBuffer) {
  return BuiltinTypeDeclBuilder(Decl)
      .addHandleMember(S, RC, RK, IsROV, RawBuffer)
      .addDefaultHandleConstructor(S, RC);
}

BinaryOperator *constructSizeOfLEQ16Expr(ASTContext &Context,
                                         SourceLocation NameLoc,
                                         TemplateTypeParmDecl *T) {
  // Obtain the QualType for 'unsigned long'
  QualType UnsignedLongType = Context.UnsignedLongTy;

  // Create a QualType that points to this TemplateTypeParmDecl
  QualType TType = Context.getTypeDeclType(T);

  // Create a TypeSourceInfo for the template type parameter 'T'
  TypeSourceInfo *TTypeSourceInfo =
      Context.getTrivialTypeSourceInfo(TType, NameLoc);

  UnaryExprOrTypeTraitExpr *sizeOfExpr = new (Context) UnaryExprOrTypeTraitExpr(
      UETT_SizeOf, TTypeSourceInfo, UnsignedLongType, NameLoc, NameLoc);

  // Create an IntegerLiteral for the value '16' with size type
  QualType SizeType = Context.getSizeType();
  llvm::APInt SizeValue = llvm::APInt(Context.getTypeSize(SizeType), 16);
  IntegerLiteral *SizeLiteral =
      new (Context) IntegerLiteral(Context, SizeValue, SizeType, NameLoc);

  QualType BoolTy = Context.BoolTy;

  BinaryOperator *binaryOperator =
      BinaryOperator::Create(Context, sizeOfExpr, // Left-hand side expression
                             SizeLiteral,         // Right-hand side expression
                             BO_LE,               // Binary operator kind (<=)
                             BoolTy,              // Result type (bool)
                             VK_LValue,           // Value kind
                             OK_Ordinary,         // Object kind
                             NameLoc,             // Source location of operator
                             FPOptionsOverride());

  return binaryOperator;
}

Expr *constructTypedBufferConstraintExpr(Sema &S, SourceLocation NameLoc,
                                         TemplateTypeParmDecl *T) {
  ASTContext &Context = S.getASTContext();

  // first get the "sizeof(T) <= 16" expression, as a binary operator
  BinaryOperator *SizeOfLEQ16 = constructSizeOfLEQ16Expr(Context, NameLoc, T);
  // TODO: add the 'builtin_hlsl_is_typed_resource_element_compatible' builtin
  // and return a binary operator that evaluates the builtin on the given
  // template type parameter 'T'.
  // Defined in issue https://github.com/llvm/llvm-project/issues/113223
  return SizeOfLEQ16;
}

ConceptDecl *constructTypedBufferConceptDecl(Sema &S, NamespaceDecl *NSD) {
  ASTContext &Context = S.getASTContext();
  DeclContext *DC = NSD->getDeclContext();
  SourceLocation DeclLoc = SourceLocation();

  IdentifierInfo &IsTypedResourceElementCompatibleII =
      Context.Idents.get("__is_typed_resource_element_compatible");
  IdentifierInfo &ElementTypeII = Context.Idents.get("element_type");
  TemplateTypeParmDecl *T = TemplateTypeParmDecl::Create(
      Context, Context.getTranslationUnitDecl(), DeclLoc, DeclLoc,
      /*depth=*/0,
      /*position=*/0,
      /*id=*/&ElementTypeII,
      /*Typename=*/true,
      /*ParameterPack=*/false);

  T->setDeclContext(DC);
  T->setReferenced();

  // Create and Attach Template Parameter List to ConceptDecl
  TemplateParameterList *ConceptParams = TemplateParameterList::Create(
      Context, DeclLoc, DeclLoc, {T}, DeclLoc, nullptr);

  DeclarationName DeclName =
      DeclarationName(&IsTypedResourceElementCompatibleII);
  Expr *ConstraintExpr = constructTypedBufferConstraintExpr(S, DeclLoc, T);

  // Create a ConceptDecl
  ConceptDecl *CD =
      ConceptDecl::Create(Context, Context.getTranslationUnitDecl(), DeclLoc,
                          DeclName, ConceptParams, ConstraintExpr);

  // Attach the template parameter list to the ConceptDecl
  CD->setTemplateParameters(ConceptParams);

  // Add the concept declaration to the Translation Unit Decl
  Context.getTranslationUnitDecl()->addDecl(CD);

  return CD;
}

void HLSLExternalSemaSource::defineHLSLTypesWithForwardDeclarations() {
  CXXRecordDecl *Decl;
  ConceptDecl *TypeBufferConcept =
      constructTypedBufferConceptDecl(*SemaPtr, HLSLNamespace);

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWBuffer")
             .addSimpleTemplateParams(*SemaPtr, {"element_type"},
                                      TypeBufferConcept)
             .Record;

  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV,
                    ResourceKind::TypedBuffer,
                    /*IsROV=*/false, /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RasterizerOrderedBuffer")
          .addSimpleTemplateParams(*SemaPtr, {"element_type"})
          .Record;
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV,
                    ResourceKind::TypedBuffer, /*IsROV=*/true,
                    /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "StructuredBuffer")
             .addSimpleTemplateParams(*SemaPtr, {"element_type"})
             .Record;
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV,
                    ResourceKind::TypedBuffer, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .completeDefinition();
  });
}

void HLSLExternalSemaSource::onCompletion(CXXRecordDecl *Record,
                                          CompletionFunction Fn) {
  Completions.insert(std::make_pair(Record->getCanonicalDecl(), Fn));
}

void HLSLExternalSemaSource::CompleteType(TagDecl *Tag) {
  if (!isa<CXXRecordDecl>(Tag))
    return;
  auto Record = cast<CXXRecordDecl>(Tag);

  // If this is a specialization, we need to get the underlying templated
  // declaration and complete that.
  if (auto TDecl = dyn_cast<ClassTemplateSpecializationDecl>(Record))
    Record = TDecl->getSpecializedTemplate()->getTemplatedDecl();
  Record = Record->getCanonicalDecl();
  auto It = Completions.find(Record);
  if (It == Completions.end())
    return;
  It->second(Record);
}
