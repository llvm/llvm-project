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
#include "HLSLBuiltinTypeDeclBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
using namespace llvm::hlsl;

using clang::hlsl::BuiltinTypeDeclBuilder;

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
                                              ResourceClass RC, bool IsROV,
                                              bool RawBuffer) {
  return BuiltinTypeDeclBuilder(S, Decl)
      .addHandleMember(RC, IsROV, RawBuffer)
      .addDefaultHandleConstructor()
      .addHandleConstructorFromBinding()
      .addHandleConstructorFromImplicitBinding();
}

// This function is responsible for constructing the constraint expression for
// this concept:
// template<typename T> concept is_typed_resource_element_compatible =
// __is_typed_resource_element_compatible<T>;
static Expr *constructTypedBufferConstraintExpr(Sema &S, SourceLocation NameLoc,
                                                TemplateTypeParmDecl *T) {
  ASTContext &Context = S.getASTContext();

  // Obtain the QualType for 'bool'
  QualType BoolTy = Context.BoolTy;

  // Create a QualType that points to this TemplateTypeParmDecl
  QualType TType = Context.getTypeDeclType(T);

  // Create a TypeSourceInfo for the template type parameter 'T'
  TypeSourceInfo *TTypeSourceInfo =
      Context.getTrivialTypeSourceInfo(TType, NameLoc);

  TypeTraitExpr *TypedResExpr = TypeTraitExpr::Create(
      Context, BoolTy, NameLoc, UTT_IsTypedResourceElementCompatible,
      {TTypeSourceInfo}, NameLoc, true);

  return TypedResExpr;
}

// This function is responsible for constructing the constraint expression for
// this concept:
// template<typename T> concept is_structured_resource_element_compatible =
// !__is_intangible<T> && sizeof(T) >= 1;
static Expr *constructStructuredBufferConstraintExpr(Sema &S,
                                                     SourceLocation NameLoc,
                                                     TemplateTypeParmDecl *T) {
  ASTContext &Context = S.getASTContext();

  // Obtain the QualType for 'bool'
  QualType BoolTy = Context.BoolTy;

  // Create a QualType that points to this TemplateTypeParmDecl
  QualType TType = Context.getTypeDeclType(T);

  // Create a TypeSourceInfo for the template type parameter 'T'
  TypeSourceInfo *TTypeSourceInfo =
      Context.getTrivialTypeSourceInfo(TType, NameLoc);

  TypeTraitExpr *IsIntangibleExpr =
      TypeTraitExpr::Create(Context, BoolTy, NameLoc, UTT_IsIntangibleType,
                            {TTypeSourceInfo}, NameLoc, true);

  // negate IsIntangibleExpr
  UnaryOperator *NotIntangibleExpr = UnaryOperator::Create(
      Context, IsIntangibleExpr, UO_LNot, BoolTy, VK_LValue, OK_Ordinary,
      NameLoc, false, FPOptionsOverride());

  // element types also may not be of 0 size
  UnaryExprOrTypeTraitExpr *SizeOfExpr = new (Context) UnaryExprOrTypeTraitExpr(
      UETT_SizeOf, TTypeSourceInfo, BoolTy, NameLoc, NameLoc);

  // Create a BinaryOperator that checks if the size of the type is not equal to
  // 1 Empty structs have a size of 1 in HLSL, so we need to check for that
  IntegerLiteral *rhs = IntegerLiteral::Create(
      Context, llvm::APInt(Context.getTypeSize(Context.getSizeType()), 1, true),
      Context.getSizeType(), NameLoc);

  BinaryOperator *SizeGEQOneExpr =
      BinaryOperator::Create(Context, SizeOfExpr, rhs, BO_GE, BoolTy, VK_LValue,
                             OK_Ordinary, NameLoc, FPOptionsOverride());

  // Combine the two constraints
  BinaryOperator *CombinedExpr = BinaryOperator::Create(
      Context, NotIntangibleExpr, SizeGEQOneExpr, BO_LAnd, BoolTy, VK_LValue,
      OK_Ordinary, NameLoc, FPOptionsOverride());

  return CombinedExpr;
}

static ConceptDecl *constructBufferConceptDecl(Sema &S, NamespaceDecl *NSD,
                                               bool isTypedBuffer) {
  ASTContext &Context = S.getASTContext();
  DeclContext *DC = NSD->getDeclContext();
  SourceLocation DeclLoc = SourceLocation();

  IdentifierInfo &ElementTypeII = Context.Idents.get("element_type");
  TemplateTypeParmDecl *T = TemplateTypeParmDecl::Create(
      Context, NSD->getDeclContext(), DeclLoc, DeclLoc,
      /*D=*/0,
      /*P=*/0,
      /*Id=*/&ElementTypeII,
      /*Typename=*/true,
      /*ParameterPack=*/false);

  T->setDeclContext(DC);
  T->setReferenced();

  // Create and Attach Template Parameter List to ConceptDecl
  TemplateParameterList *ConceptParams = TemplateParameterList::Create(
      Context, DeclLoc, DeclLoc, {T}, DeclLoc, nullptr);

  DeclarationName DeclName;
  Expr *ConstraintExpr = nullptr;

  if (isTypedBuffer) {
    DeclName = DeclarationName(
        &Context.Idents.get("__is_typed_resource_element_compatible"));
    ConstraintExpr = constructTypedBufferConstraintExpr(S, DeclLoc, T);
  } else {
    DeclName = DeclarationName(
        &Context.Idents.get("__is_structured_resource_element_compatible"));
    ConstraintExpr = constructStructuredBufferConstraintExpr(S, DeclLoc, T);
  }

  // Create a ConceptDecl
  ConceptDecl *CD =
      ConceptDecl::Create(Context, NSD->getDeclContext(), DeclLoc, DeclName,
                          ConceptParams, ConstraintExpr);

  // Attach the template parameter list to the ConceptDecl
  CD->setTemplateParameters(ConceptParams);

  // Add the concept declaration to the Translation Unit Decl
  NSD->getDeclContext()->addDecl(CD);

  return CD;
}

void HLSLExternalSemaSource::defineHLSLTypesWithForwardDeclarations() {
  CXXRecordDecl *Decl;
  ConceptDecl *TypedBufferConcept = constructBufferConceptDecl(
      *SemaPtr, HLSLNamespace, /*isTypedBuffer*/ true);
  ConceptDecl *StructuredBufferConcept = constructBufferConceptDecl(
      *SemaPtr, HLSLNamespace, /*isTypedBuffer*/ false);

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "Buffer")
             .addSimpleTemplateParams({"element_type"}, TypedBufferConcept)
             .finalizeForwardDeclaration();

  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, /*IsROV=*/false,
                    /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWBuffer")
             .addSimpleTemplateParams({"element_type"}, TypedBufferConcept)
             .finalizeForwardDeclaration();

  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RasterizerOrderedBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/true,
                    /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "StructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWStructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addIncrementCounterMethod()
        .addDecrementCounterMethod()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "AppendStructuredBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .addAppendMethod()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "ConsumeStructuredBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .addConsumeMethod()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace,
                                "RasterizerOrderedStructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/true,
                    /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addIncrementCounterMethod()
        .addDecrementCounterMethod()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "ByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .completeDefinition();
  });
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .completeDefinition();
  });
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace,
                                "RasterizerOrderedByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/true,
                    /*RawBuffer=*/true)
        .completeDefinition();
  });
}

void HLSLExternalSemaSource::onCompletion(CXXRecordDecl *Record,
                                          CompletionFunction Fn) {
  if (!Record->isCompleteDefinition())
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
