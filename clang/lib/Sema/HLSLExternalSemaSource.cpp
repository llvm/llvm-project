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
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Sema/Sema.h"

#include <functional>

using namespace clang;

namespace {

struct TemplateParameterListBuilder;

struct BuiltinTypeDeclBuilder {
  CXXRecordDecl *Record = nullptr;
  ClassTemplateDecl *Template = nullptr;
  NamespaceDecl *HLSLNamespace = nullptr;

  BuiltinTypeDeclBuilder(CXXRecordDecl *R) : Record(R) {
    Record->startDefinition();
    Template = Record->getDescribedClassTemplate();
  }

  BuiltinTypeDeclBuilder(Sema &S, NamespaceDecl *Namespace, StringRef Name)
      : HLSLNamespace(Namespace) {
    ASTContext &AST = S.getASTContext();
    IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);

    Record = CXXRecordDecl::Create(AST, TagDecl::TagKind::TTK_Class,
                                   HLSLNamespace, SourceLocation(),
                                   SourceLocation(), &II, nullptr, true);
    Record->setImplicit(true);
    Record->setLexicalDeclContext(HLSLNamespace);
    Record->setHasExternalLexicalStorage();

    // Don't let anyone derive from built-in types
    Record->addAttr(FinalAttr::CreateImplicit(AST, SourceRange(),
                                              AttributeCommonInfo::AS_Keyword,
                                              FinalAttr::Keyword_final));
  }

  ~BuiltinTypeDeclBuilder() {
    if (HLSLNamespace && !Template)
      HLSLNamespace->addDecl(Record);
  }

  BuiltinTypeDeclBuilder &
  addTemplateArgumentList(llvm::ArrayRef<NamedDecl *> TemplateArgs) {
    ASTContext &AST = Record->getASTContext();

    auto *ParamList =
        TemplateParameterList::Create(AST, SourceLocation(), SourceLocation(),
                                      TemplateArgs, SourceLocation(), nullptr);
    Template = ClassTemplateDecl::Create(
        AST, Record->getDeclContext(), SourceLocation(),
        DeclarationName(Record->getIdentifier()), ParamList, Record);
    Record->setDescribedClassTemplate(Template);
    Template->setImplicit(true);
    Template->setLexicalDeclContext(Record->getDeclContext());
    Record->getDeclContext()->addDecl(Template);

    // Requesting the class name specialization will fault in required types.
    QualType T = Template->getInjectedClassNameSpecialization();
    T = AST.getInjectedClassNameType(Record, T);
    return *this;
  }

  BuiltinTypeDeclBuilder &
  addMemberVariable(StringRef Name, QualType Type,
                    AccessSpecifier Access = AccessSpecifier::AS_private) {
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
    Record->addDecl(Field);
    return *this;
  }

  BuiltinTypeDeclBuilder &
  addHandleMember(AccessSpecifier Access = AccessSpecifier::AS_private) {
    return addMemberVariable("h", Record->getASTContext().VoidPtrTy, Access);
  }

  BuiltinTypeDeclBuilder &startDefinition() {
    Record->startDefinition();
    return *this;
  }

  BuiltinTypeDeclBuilder &completeDefinition() {
    assert(Record->isBeingDefined() &&
           "Definition must be started before completing it.");

    Record->completeDefinition();
    return *this;
  }

  TemplateParameterListBuilder addTemplateArgumentList();
};

struct TemplateParameterListBuilder {
  BuiltinTypeDeclBuilder &Builder;
  ASTContext &AST;
  llvm::SmallVector<NamedDecl *> Params;

  TemplateParameterListBuilder(BuiltinTypeDeclBuilder &RB)
      : Builder(RB), AST(RB.Record->getASTContext()) {}

  ~TemplateParameterListBuilder() { finalizeTemplateArgs(); }

  TemplateParameterListBuilder &
  addTypeParameter(StringRef Name, QualType DefaultValue = QualType()) {
    unsigned Position = static_cast<unsigned>(Params.size());
    auto *Decl = TemplateTypeParmDecl::Create(
        AST, Builder.Record->getDeclContext(), SourceLocation(),
        SourceLocation(), /* TemplateDepth */ 0, Position,
        &AST.Idents.get(Name, tok::TokenKind::identifier), /* Typename */ false,
        /* ParameterPack */ false);
    if (!DefaultValue.isNull())
      Decl->setDefaultArgument(AST.getTrivialTypeSourceInfo(DefaultValue));

    Params.emplace_back(Decl);
    return *this;
  }

  BuiltinTypeDeclBuilder &finalizeTemplateArgs() {
    if (Params.empty())
      return Builder;
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
    Builder.Record->getDeclContext()->addDecl(Builder.Template);
    Params.clear();

    QualType T = Builder.Template->getInjectedClassNameSpecialization();
    T = AST.getInjectedClassNameType(Builder.Record, T);

    return Builder;
  }
};

TemplateParameterListBuilder BuiltinTypeDeclBuilder::addTemplateArgumentList() {
  return TemplateParameterListBuilder(*this);
}
} // namespace

HLSLExternalSemaSource::~HLSLExternalSemaSource() {}

void HLSLExternalSemaSource::InitializeSema(Sema &S) {
  SemaPtr = &S;
  ASTContext &AST = SemaPtr->getASTContext();
  IdentifierInfo &HLSL = AST.Idents.get("hlsl", tok::TokenKind::identifier);
  HLSLNamespace =
      NamespaceDecl::Create(AST, AST.getTranslationUnitDecl(), false,
                            SourceLocation(), SourceLocation(), &HLSL, nullptr);
  HLSLNamespace->setImplicit(true);
  AST.getTranslationUnitDecl()->addDecl(HLSLNamespace);
  defineTrivialHLSLTypes();
  forwardDeclareHLSLTypes();

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
  TypeParam->setDefaultArgument(AST.getTrivialTypeSourceInfo(AST.FloatTy));

  TemplateParams.emplace_back(TypeParam);

  auto *SizeParam = NonTypeTemplateParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 1,
      &AST.Idents.get("element_count", tok::TokenKind::identifier), AST.IntTy,
      false, AST.getTrivialTypeSourceInfo(AST.IntTy));
  Expr *LiteralExpr =
      IntegerLiteral::Create(AST, llvm::APInt(AST.getIntWidth(AST.IntTy), 4),
                             AST.IntTy, SourceLocation());
  SizeParam->setDefaultArgument(LiteralExpr);
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

  ResourceDecl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "Resource")
                     .startDefinition()
                     .addHandleMember(AccessSpecifier::AS_public)
                     .completeDefinition()
                     .Record;
}

void HLSLExternalSemaSource::forwardDeclareHLSLTypes() {
  CXXRecordDecl *Decl;
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWBuffer")
             .addTemplateArgumentList()
             .addTypeParameter("element_type", SemaPtr->getASTContext().FloatTy)
             .finalizeTemplateArgs()
             .Record;
  Completions.insert(std::make_pair(
      Decl, std::bind(&HLSLExternalSemaSource::completeBufferType, this,
                      std::placeholders::_1)));
}

void HLSLExternalSemaSource::CompleteType(TagDecl *Tag) {
  if (!isa<CXXRecordDecl>(Tag))
    return;
  auto Record = cast<CXXRecordDecl>(Tag);

  // If this is a specialization, we need to get the underlying templated
  // declaration and complete that.
  if (auto TDecl = dyn_cast<ClassTemplateSpecializationDecl>(Record))
    Record = TDecl->getSpecializedTemplate()->getTemplatedDecl();
  auto It = Completions.find(Record);
  if (It == Completions.end())
    return;
  It->second(Record);
}

void HLSLExternalSemaSource::completeBufferType(CXXRecordDecl *Record) {
  BuiltinTypeDeclBuilder(Record).addHandleMember().completeDefinition();
}
