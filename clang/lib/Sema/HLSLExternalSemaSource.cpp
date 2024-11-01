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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/HLSLRuntime.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/Support/ErrorHandling.h"

#include <functional>

using namespace clang;
using namespace llvm::hlsl;

static FunctionDecl *lookupBuiltinFunction(Sema &S, StringRef Name);

namespace {

struct TemplateParameterListBuilder;

struct BuiltinTypeDeclBuilder {
  Sema &S;
  CXXRecordDecl *Record = nullptr;
  ClassTemplateDecl *Template = nullptr;
  ClassTemplateDecl *PrevTemplate = nullptr;
  NamespaceDecl *HLSLNamespace = nullptr;
  llvm::StringMap<FieldDecl *> Fields;

  BuiltinTypeDeclBuilder(Sema &S, CXXRecordDecl *R) : S(S), Record(R) {
    Record->startDefinition();
    Template = Record->getDescribedClassTemplate();
  }

  BuiltinTypeDeclBuilder(Sema &S, NamespaceDecl *Namespace, StringRef Name)
      : S(S), HLSLNamespace(Namespace) {
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
  addHandleMember(ResourceClass RC, ResourceKind RK, bool IsROV, bool RawBuffer,
                  AccessSpecifier Access = AccessSpecifier::AS_private) {
    if (Record->isCompleteDefinition())
      return *this;

    ASTContext &Ctx = S.getASTContext();
    TypeSourceInfo *ElementTypeInfo = nullptr;

    QualType ElemTy = Ctx.Char8Ty;
    if (Template)
      ElemTy = getFirstTemplateTypeParam();
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

  static Expr *emitResourceClassExpr(ASTContext &AST, ResourceClass RC) {
    return IntegerLiteral::Create(
        AST,
        llvm::APInt(AST.getIntWidth(AST.UnsignedCharTy),
                    static_cast<uint8_t>(RC)),
        AST.UnsignedCharTy, SourceLocation());
  }

  BuiltinTypeDeclBuilder &addDefaultHandleConstructor(ResourceClass RC) {
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
    if (Template)
      ElemTy = getFirstTemplateTypeParam();
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

  FieldDecl *getResourceHandleField() {
    FieldDecl *FD = Fields["h"];
    assert(FD && FD->getType()->isHLSLAttributedResourceType() &&
           "record does not have resource handle");
    return FD;
  }

  QualType getFirstTemplateTypeParam() {
    assert(Template && "record it not a template");
    if (const auto *TTD = dyn_cast<TemplateTypeParmDecl>(
            Template->getTemplateParameters()->getParam(0))) {
      return QualType(TTD->getTypeForDecl(), 0);
    }
    return QualType();
  }

  BuiltinTypeDeclBuilder &startDefinition() {
    // we might already have complete definition from a precompiled header
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

  TemplateParameterListBuilder addTemplateArgumentList();
  BuiltinTypeDeclBuilder &addSimpleTemplateParams(ArrayRef<StringRef> Names);

  // Builtin types methods
  BuiltinTypeDeclBuilder &addIncrementCounterMethod();
  BuiltinTypeDeclBuilder &addDecrementCounterMethod();
};

struct TemplateParameterListBuilder {
  BuiltinTypeDeclBuilder &Builder;
  llvm::SmallVector<NamedDecl *> Params;

  TemplateParameterListBuilder(BuiltinTypeDeclBuilder &RB) : Builder(RB) {}

  ~TemplateParameterListBuilder() { finalizeTemplateArgs(); }

  TemplateParameterListBuilder &
  addTypeParameter(StringRef Name, QualType DefaultValue = QualType()) {
    if (Builder.Record->isCompleteDefinition())
      return *this;
    ASTContext &AST = Builder.S.getASTContext();
    unsigned Position = static_cast<unsigned>(Params.size());
    auto *Decl = TemplateTypeParmDecl::Create(
        AST, Builder.Record->getDeclContext(), SourceLocation(),
        SourceLocation(), /* TemplateDepth */ 0, Position,
        &AST.Idents.get(Name, tok::TokenKind::identifier),
        /* Typename */ false,
        /* ParameterPack */ false);
    if (!DefaultValue.isNull())
      Decl->setDefaultArgument(
          AST, Builder.S.getTrivialTemplateArgumentLoc(DefaultValue, QualType(),
                                                       SourceLocation()));

    Params.emplace_back(Decl);
    return *this;
  }

  BuiltinTypeDeclBuilder &finalizeTemplateArgs() {
    if (Params.empty())
      return Builder;
    ASTContext &AST = Builder.S.Context;
    auto *ParamList =
        TemplateParameterList::Create(AST, SourceLocation(), SourceLocation(),
                                      Params, SourceLocation(), nullptr);
    Builder.Template = ClassTemplateDecl::Create(
        AST, Builder.Record->getDeclContext(), SourceLocation(),
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
    T = AST.getInjectedClassNameType(Builder.Record, T);

    return Builder;
  }
};

// Builder for methods of builtin types. Allows adding methods to builtin types
// using the builder pattern like this:
//
//   BuiltinTypeMethodBuilder(Sema, RecordBuilder, "MethodName", ReturnType)
//       .addParam("param_name", Type, InOutModifier)
//       .callBuiltin("buildin_name", { BuiltinParams })
//       .finalizeMethod();
//
// The builder needs to have all of the method parameters before it can create
// a CXXMethodDecl. It collects them in addParam calls and when a first
// method that builds the body is called it creates the CXXMethodDecl and
// ParmVarDecls instances. These can then be referenced from the body building
// methods. Destructor or an explicit call to finalizeMethod() will complete
// the method definition.
struct BuiltinTypeMethodBuilder {
  struct MethodParam {
    const IdentifierInfo &NameII;
    QualType Ty;
    HLSLParamModifierAttr::Spelling Modifier;
    MethodParam(const IdentifierInfo &NameII, QualType Ty,
                HLSLParamModifierAttr::Spelling Modifier)
        : NameII(NameII), Ty(Ty), Modifier(Modifier) {}
  };

  BuiltinTypeDeclBuilder &DeclBuilder;
  DeclarationNameInfo NameInfo;
  QualType ReturnTy;
  CXXMethodDecl *Method;
  llvm::SmallVector<MethodParam> Params;
  llvm::SmallVector<Stmt *> StmtsList;

public:
  BuiltinTypeMethodBuilder(Sema &S, BuiltinTypeDeclBuilder &DB, StringRef Name,
                           QualType ReturnTy)
      : DeclBuilder(DB), ReturnTy(ReturnTy), Method(nullptr) {
    const IdentifierInfo &II =
        S.getASTContext().Idents.get(Name, tok::TokenKind::identifier);
    NameInfo = DeclarationNameInfo(DeclarationName(&II), SourceLocation());
  }

  BuiltinTypeMethodBuilder &addParam(StringRef Name, QualType Ty,
                                     HLSLParamModifierAttr::Spelling Modifier =
                                         HLSLParamModifierAttr::Keyword_in) {
    assert(Method == nullptr && "Cannot add param, method already created");

    const IdentifierInfo &II = DeclBuilder.S.getASTContext().Idents.get(
        Name, tok::TokenKind::identifier);
    Params.emplace_back(II, Ty, Modifier);
    return *this;
  }

private:
  void createMethodDecl() {
    assert(Method == nullptr && "Method already created");

    // create method type
    ASTContext &AST = DeclBuilder.S.getASTContext();
    SmallVector<QualType> ParamTypes;
    for (auto &MP : Params)
      ParamTypes.emplace_back(MP.Ty);
    QualType MethodTy = AST.getFunctionType(ReturnTy, ParamTypes,
                                            FunctionProtoType::ExtProtoInfo());

    // create method decl
    auto *TSInfo = AST.getTrivialTypeSourceInfo(MethodTy, SourceLocation());
    Method =
        CXXMethodDecl::Create(AST, DeclBuilder.Record, SourceLocation(),
                              NameInfo, MethodTy, TSInfo, SC_None, false, false,
                              ConstexprSpecKind::Unspecified, SourceLocation());

    // create params & set them to the function prototype
    SmallVector<ParmVarDecl *> ParmDecls;
    auto FnProtoLoc =
        Method->getTypeSourceInfo()->getTypeLoc().getAs<FunctionProtoTypeLoc>();
    unsigned i = 0;
    for (auto &MP : Params) {
      ParmVarDecl *Parm = ParmVarDecl::Create(
          AST, Method->getDeclContext(), SourceLocation(), SourceLocation(),
          &MP.NameII, MP.Ty,
          AST.getTrivialTypeSourceInfo(MP.Ty, SourceLocation()), SC_None,
          nullptr);
      if (MP.Modifier != HLSLParamModifierAttr::Keyword_in) {
        auto *Mod =
            HLSLParamModifierAttr::Create(AST, SourceRange(), MP.Modifier);
        Parm->addAttr(Mod);
      }
      ParmDecls.push_back(Parm);
      FnProtoLoc.setParam(i++, Parm);
    }
    Method->setParams({ParmDecls});
  }

  void addResourceHandleToParms(SmallVector<Expr *> &Parms) {
    ASTContext &AST = DeclBuilder.S.getASTContext();
    FieldDecl *HandleField = DeclBuilder.getResourceHandleField();
    auto *This = CXXThisExpr::Create(
        AST, SourceLocation(), Method->getFunctionObjectParameterType(), true);
    Parms.push_back(MemberExpr::CreateImplicit(AST, This, false, HandleField,
                                               HandleField->getType(),
                                               VK_LValue, OK_Ordinary));
  }

public:
  ~BuiltinTypeMethodBuilder() { finalizeMethod(); }

  BuiltinTypeMethodBuilder &
  callBuiltin(StringRef BuiltinName, ArrayRef<Expr *> CallParms,
              bool AddResourceHandleAsFirstArg = true) {
    if (!Method)
      createMethodDecl();

    ASTContext &AST = DeclBuilder.S.getASTContext();
    FunctionDecl *FD = lookupBuiltinFunction(DeclBuilder.S, BuiltinName);
    DeclRefExpr *DRE = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), SourceLocation(), FD, false,
        FD->getNameInfo(), FD->getType(), VK_PRValue);

    SmallVector<Expr *> NewCallParms;
    if (AddResourceHandleAsFirstArg) {
      addResourceHandleToParms(NewCallParms);
      for (auto *P : CallParms)
        NewCallParms.push_back(P);
    }

    Expr *Call = CallExpr::Create(
        AST, DRE, AddResourceHandleAsFirstArg ? NewCallParms : CallParms,
        FD->getReturnType(), VK_PRValue, SourceLocation(), FPOptionsOverride());
    StmtsList.push_back(Call);
    return *this;
  }

  BuiltinTypeMethodBuilder &
  callBuiltinForwardArgs(StringRef BuiltinName,
                         bool AddResourceHandleAsFirstArg = true) {
    // FIXME: Call the buildin with all of the method parameters
    // plus optional resource handle as the first arg.
    llvm_unreachable("not yet implemented");
  }

  BuiltinTypeDeclBuilder &finalizeMethod() {
    if (DeclBuilder.Record->isCompleteDefinition())
      return DeclBuilder;

    if (!Method)
      createMethodDecl();

    if (!Method->hasBody()) {
      ASTContext &AST = DeclBuilder.S.getASTContext();
      if (ReturnTy != AST.VoidTy && !StmtsList.empty()) {
        if (Expr *LastExpr = dyn_cast<Expr>(StmtsList.back())) {
          assert(AST.hasSameUnqualifiedType(
                     isa<CallExpr>(LastExpr)
                         ? cast<CallExpr>(LastExpr)->getCallReturnType(AST)
                         : LastExpr->getType(),
                     ReturnTy) &&
                 "Return type of the last statement must match the return type "
                 "of the method");
          StmtsList.pop_back();
          StmtsList.push_back(
              ReturnStmt::Create(AST, SourceLocation(), LastExpr, nullptr));
        }
      }

      Method->setBody(CompoundStmt::Create(AST, StmtsList, FPOptionsOverride(),
                                           SourceLocation(), SourceLocation()));
      Method->setLexicalDeclContext(DeclBuilder.Record);
      Method->setAccess(AccessSpecifier::AS_public);
      Method->addAttr(AlwaysInlineAttr::CreateImplicit(
          AST, SourceRange(), AlwaysInlineAttr::CXX11_clang_always_inline));
      DeclBuilder.Record->addDecl(Method);
    }
    return DeclBuilder;
  }
};

} // namespace

TemplateParameterListBuilder BuiltinTypeDeclBuilder::addTemplateArgumentList() {
  return TemplateParameterListBuilder(*this);
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSimpleTemplateParams(ArrayRef<StringRef> Names) {
  TemplateParameterListBuilder Builder = this->addTemplateArgumentList();
  for (StringRef Name : Names)
    Builder.addTypeParameter(Name);
  return Builder.finalizeTemplateArgs();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addIncrementCounterMethod() {
  ASTContext &AST = S.getASTContext();
  Expr *One =
      IntegerLiteral::Create(AST, llvm::APInt(AST.getTypeSize(AST.IntTy), 1),
                             AST.IntTy, SourceLocation());
  return BuiltinTypeMethodBuilder(S, *this, "IncrementCounter",
                                  AST.UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", {One})
      .finalizeMethod();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addDecrementCounterMethod() {
  ASTContext &AST = S.getASTContext();
  Expr *NegOne =
      IntegerLiteral::Create(AST, llvm::APInt(AST.getTypeSize(AST.IntTy), -1),
                             AST.IntTy, SourceLocation());
  return BuiltinTypeMethodBuilder(S, *this, "DecrementCounter",
                                  AST.UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", {NegOne})
      .finalizeMethod();
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
  return BuiltinTypeDeclBuilder(S, Decl)
      .addHandleMember(RC, RK, IsROV, RawBuffer)
      .addDefaultHandleConstructor(RC);
}

void HLSLExternalSemaSource::defineHLSLTypesWithForwardDeclarations() {
  CXXRecordDecl *Decl;
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWBuffer")
             .addSimpleTemplateParams({"element_type"})
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
          .addSimpleTemplateParams({"element_type"})
          .Record;
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV,
                    ResourceKind::TypedBuffer, /*IsROV=*/true,
                    /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "StructuredBuffer")
             .addSimpleTemplateParams({"element_type"})
             .Record;
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV,
                    ResourceKind::TypedBuffer, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWStructuredBuffer")
             .addSimpleTemplateParams({"element_type"})
             .Record;
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV,
                    ResourceKind::TypedBuffer, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .addIncrementCounterMethod()
        .addDecrementCounterMethod()
        .completeDefinition();
  });

  // FIXME: Also add Increment/DecrementCounter to
  // RasterizerOrderedStructuredBuffer when llvm/llvm-project/#113648 is merged.
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

static FunctionDecl *lookupBuiltinFunction(Sema &S, StringRef Name) {
  IdentifierInfo &II =
      S.getASTContext().Idents.get(Name, tok::TokenKind::identifier);
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
  assert(isa<FunctionDecl>(R.getFoundDecl()));
  return cast<FunctionDecl>(R.getFoundDecl());
}
