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
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
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
  defineHLSLAtomicIntrinsics();

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
  SizeParam->setDefaultArgument(AST, SemaPtr->getTrivialTemplateArgumentLoc(
                                         Default, AST.IntTy, SourceLocation()));
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

void HLSLExternalSemaSource::defineHLSLMatrixAlias() {
  ASTContext &AST = SemaPtr->getASTContext();
  llvm::SmallVector<NamedDecl *> TemplateParams;

  auto *TypeParam = TemplateTypeParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 0,
      &AST.Idents.get("element", tok::TokenKind::identifier), false, false);
  TypeParam->setDefaultArgument(
      AST, SemaPtr->getTrivialTemplateArgumentLoc(
               TemplateArgument(AST.FloatTy), QualType(), SourceLocation()));

  TemplateParams.emplace_back(TypeParam);

  // these should be 64 bit to be consistent with other clang matrices.
  auto *RowsParam = NonTypeTemplateParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 1,
      &AST.Idents.get("rows_count", tok::TokenKind::identifier), AST.IntTy,
      false, AST.getTrivialTypeSourceInfo(AST.IntTy));
  llvm::APInt RVal(AST.getIntWidth(AST.IntTy), 4);
  TemplateArgument RDefault(AST, llvm::APSInt(std::move(RVal)), AST.IntTy,
                            /*IsDefaulted=*/true);
  RowsParam->setDefaultArgument(
      AST, SemaPtr->getTrivialTemplateArgumentLoc(RDefault, AST.IntTy,
                                                  SourceLocation()));
  TemplateParams.emplace_back(RowsParam);

  auto *ColsParam = NonTypeTemplateParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 2,
      &AST.Idents.get("cols_count", tok::TokenKind::identifier), AST.IntTy,
      false, AST.getTrivialTypeSourceInfo(AST.IntTy));
  llvm::APInt CVal(AST.getIntWidth(AST.IntTy), 4);
  TemplateArgument CDefault(AST, llvm::APSInt(std::move(CVal)), AST.IntTy,
                            /*IsDefaulted=*/true);
  ColsParam->setDefaultArgument(
      AST, SemaPtr->getTrivialTemplateArgumentLoc(CDefault, AST.IntTy,
                                                  SourceLocation()));
  TemplateParams.emplace_back(ColsParam);

  const unsigned MaxMatDim = SemaPtr->getLangOpts().MaxMatrixDimension;

  auto *MaxRow = IntegerLiteral::Create(
      AST, llvm::APInt(AST.getIntWidth(AST.IntTy), MaxMatDim), AST.IntTy,
      SourceLocation());
  auto *MaxCol = IntegerLiteral::Create(
      AST, llvm::APInt(AST.getIntWidth(AST.IntTy), MaxMatDim), AST.IntTy,
      SourceLocation());

  auto *RowsRef = DeclRefExpr::Create(
      AST, NestedNameSpecifierLoc(), SourceLocation(), RowsParam,
      /*RefersToEnclosingVariableOrCapture*/ false,
      DeclarationNameInfo(RowsParam->getDeclName(), SourceLocation()),
      AST.IntTy, VK_LValue);
  auto *ColsRef = DeclRefExpr::Create(
      AST, NestedNameSpecifierLoc(), SourceLocation(), ColsParam,
      /*RefersToEnclosingVariableOrCapture*/ false,
      DeclarationNameInfo(ColsParam->getDeclName(), SourceLocation()),
      AST.IntTy, VK_LValue);

  auto *RowsLE = BinaryOperator::Create(AST, RowsRef, MaxRow, BO_LE, AST.BoolTy,
                                        VK_PRValue, OK_Ordinary,
                                        SourceLocation(), FPOptionsOverride());
  auto *ColsLE = BinaryOperator::Create(AST, ColsRef, MaxCol, BO_LE, AST.BoolTy,
                                        VK_PRValue, OK_Ordinary,
                                        SourceLocation(), FPOptionsOverride());

  auto *RequiresExpr = BinaryOperator::Create(
      AST, RowsLE, ColsLE, BO_LAnd, AST.BoolTy, VK_PRValue, OK_Ordinary,
      SourceLocation(), FPOptionsOverride());

  auto *ParamList = TemplateParameterList::Create(
      AST, SourceLocation(), SourceLocation(), TemplateParams, SourceLocation(),
      RequiresExpr);

  IdentifierInfo &II = AST.Idents.get("matrix", tok::TokenKind::identifier);

  QualType AliasType = AST.getDependentSizedMatrixType(
      AST.getTemplateTypeParmType(0, 0, false, TypeParam),
      DeclRefExpr::Create(
          AST, NestedNameSpecifierLoc(), SourceLocation(), RowsParam, false,
          DeclarationNameInfo(RowsParam->getDeclName(), SourceLocation()),
          AST.IntTy, VK_LValue),
      DeclRefExpr::Create(
          AST, NestedNameSpecifierLoc(), SourceLocation(), ColsParam, false,
          DeclarationNameInfo(ColsParam->getDeclName(), SourceLocation()),
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
  defineHLSLMatrixAlias();
}

/// Set up common members and attributes for buffer types
static BuiltinTypeDeclBuilder setupBufferType(CXXRecordDecl *Decl, Sema &S,
                                              ResourceClass RC, bool IsROV,
                                              bool RawBuffer, bool HasCounter) {
  return BuiltinTypeDeclBuilder(S, Decl)
      .addBufferHandles(RC, IsROV, RawBuffer, HasCounter)
      .addDefaultHandleConstructor()
      .addCopyConstructor()
      .addCopyAssignmentOperator()
      .addStaticInitializationFunctions(HasCounter);
}

/// Set up common members and attributes for sampler types
static BuiltinTypeDeclBuilder setupSamplerType(CXXRecordDecl *Decl, Sema &S) {
  return BuiltinTypeDeclBuilder(S, Decl)
      .addSamplerHandle()
      .addDefaultHandleConstructor()
      .addCopyConstructor()
      .addCopyAssignmentOperator()
      .addStaticInitializationFunctions(false);
}

namespace {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Which members a texture type has. Overloads within a member family
/// (e.g., offset overloads for samplers) follow from ResourceDimension.
enum class TexCap : uint32_t {
  Load = 1u << 0,      // Load(int<N+1>) taking a mip level
  LoadMS = 1u << 1,    // Load(int<N>, int sampleIndex) on a multisampled type
  LoadRW = 1u << 2,    // Load(int<N>) on a writable texture
  Subscript = 1u << 3, // operator[]
  Mips = 1u << 4,      // mips[]
  Sample = 1u << 5,    // Sample, SampleBias, SampleGrad, SampleLevel
  SampleCmp = 1u << 6, // SampleCmp, SampleCmpLevelZero
  Gather = 1u << 7,    // Gather*, GatherCmp*
  CalcLOD = 1u << 8,   // CalculateLevelOfDetail, ...Unclamped
  GetDims = 1u << 9,   // GetDimensions

  // TODO: multisampled types need an MS-specific GetDimensions
  // https://github.com/llvm/wg-hlsl/issues/347

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/GetDims)
};

/// How a type's template parameters are spelled. Independent of its
/// capabilities; also decides which types get a vector partial specialization.
enum class TemplateShape {
  ElementType,               // template<typename T = float4>
  ElementTypeAndSampleCount, // template<typename T, uint N>
};

struct TextureTypeInfo {
  const char *Name;
  ResourceClass RC;
  ResourceDimension Dim;
  bool IsArray;
  bool IsROV;
  TemplateShape Shape;
  TexCap Caps;

  bool has(TexCap C) const { return (Caps & C) != TexCap{}; }
  bool hasSampleCount() const {
    return Shape == TemplateShape::ElementTypeAndSampleCount;
  }
};
} // namespace

static const TextureTypeInfo TextureTypes[] = {
    {"Texture2D", ResourceClass::SRV, ResourceDimension::Dim2D,
     /*IsArray=*/false, /*IsROV=*/false, TemplateShape::ElementType,
     TexCap::Load | TexCap::Subscript | TexCap::Mips | TexCap::Sample |
         TexCap::SampleCmp | TexCap::CalcLOD | TexCap::Gather |
         TexCap::GetDims},
    {"RWTexture2D", ResourceClass::UAV, ResourceDimension::Dim2D,
     /*IsArray=*/false, /*IsROV=*/false, TemplateShape::ElementType,
     TexCap::LoadRW | TexCap::Subscript | TexCap::GetDims},
    {"Texture2DArray", ResourceClass::SRV, ResourceDimension::Dim2D,
     /*IsArray=*/true, /*IsROV=*/false, TemplateShape::ElementType,
     TexCap::Load | TexCap::Subscript | TexCap::Mips | TexCap::Sample |
         TexCap::SampleCmp | TexCap::CalcLOD | TexCap::Gather |
         TexCap::GetDims},
    {"RWTexture2DArray", ResourceClass::UAV, ResourceDimension::Dim2D,
     /*IsArray=*/true, /*IsROV=*/false, TemplateShape::ElementType,
     TexCap::LoadRW | TexCap::Subscript | TexCap::GetDims},
    {"Texture2DMS", ResourceClass::SRV, ResourceDimension::Dim2D,
     /*IsArray=*/false, /*IsROV=*/false,
     TemplateShape::ElementTypeAndSampleCount,
     TexCap::LoadMS | TexCap::Subscript},
    {"TextureCube", ResourceClass::SRV, ResourceDimension::Cube,
     /*IsArray=*/false, /*IsROV=*/false, TemplateShape::ElementType,
     TexCap::Sample | TexCap::SampleCmp | TexCap::CalcLOD | TexCap::Gather |
         TexCap::GetDims},
    {"TextureCubeArray", ResourceClass::SRV, ResourceDimension::Cube,
     /*IsArray=*/true, /*IsROV=*/false, TemplateShape::ElementType,
     TexCap::Sample | TexCap::SampleCmp | TexCap::CalcLOD | TexCap::Gather |
         TexCap::GetDims},
};

static BuiltinTypeDeclBuilder setupTextureType(CXXRecordDecl *Decl, Sema &S,
                                               const TextureTypeInfo &T) {
  const ResourceDimension Dim = T.Dim;
  const bool IsArray = T.IsArray;

  Expr *SampleCountExpr = nullptr;
  if (T.hasSampleCount()) {
    ClassTemplateDecl *CTD = Decl->getDescribedClassTemplate();
    assert(CTD && "multisampled texture must be a class template");
    // Parameter 1 is the N in Texture2DMS<T, N>.
    auto *NTTP = cast<NonTypeTemplateParmDecl>(
        CTD->getTemplateParameters()->getParam(1));
    SampleCountExpr =
        S.BuildDeclRefExpr(NTTP, NTTP->getType(), VK_PRValue, SourceLocation());
  }

  BuiltinTypeDeclBuilder B(S, Decl);
  B.addTextureHandle(T.RC, T.IsROV, IsArray, Dim, SampleCountExpr);

  // The `mips` member holds a second copy of the resource handle.
  // addCopyConstructor, addCopyAssignmentOperator and
  // addStaticInitializationFunctions are what initialize that copy, and they
  // look the member up by name, so it has to exist before they run.
  if (T.has(TexCap::Mips))
    B.addMipsMember(Dim);

  B.addDefaultHandleConstructor()
      .addCopyConstructor()
      .addCopyAssignmentOperator()
      .addStaticInitializationFunctions(false);

  if (T.has(TexCap::Load))
    B.addTextureLoadMethods(Dim, IsArray);
  if (T.has(TexCap::LoadMS))
    B.addTextureLoadMSMethods(Dim, IsArray);
  if (T.has(TexCap::LoadRW))
    B.addRWTextureLoadMethods(Dim, IsArray);
  if (T.has(TexCap::Subscript))
    B.addArraySubscriptOperators(Dim, IsArray);

  if (T.has(TexCap::Sample))
    B.addSampleMethods(Dim, IsArray)
        .addSampleBiasMethods(Dim, IsArray)
        .addSampleGradMethods(Dim, IsArray)
        .addSampleLevelMethods(Dim, IsArray);
  if (T.has(TexCap::SampleCmp))
    B.addSampleCmpMethods(Dim, IsArray)
        .addSampleCmpLevelZeroMethods(Dim, IsArray);
  if (T.has(TexCap::CalcLOD))
    B.addCalculateLodMethods(Dim);
  if (T.has(TexCap::GetDims))
    B.addGetDimensionsMethods(Dim);
  if (T.has(TexCap::Gather))
    B.addGatherMethods(Dim, IsArray).addGatherCmpMethods(Dim, IsArray);

  return B;
}

// Add a partial specialization for a template. The `TextureTemplate` is
// `Texture<element_type>`, and it will be specialized for vectors:
// `Texture<vector<element_type, element_count>>`.
static ClassTemplatePartialSpecializationDecl *
addVectorTexturePartialSpecialization(Sema &S, NamespaceDecl *HLSLNamespace,
                                      ClassTemplateDecl *TextureTemplate) {
  ASTContext &AST = S.getASTContext();

  // Create the template parameters: element_type and element_count.
  auto *ElementType = TemplateTypeParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 0,
      &AST.Idents.get("element_type"), false, false);
  auto *ElementCount = NonTypeTemplateParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 1,
      &AST.Idents.get("element_count"), AST.IntTy, false,
      AST.getTrivialTypeSourceInfo(AST.IntTy));

  auto *TemplateParams = TemplateParameterList::Create(
      AST, SourceLocation(), SourceLocation(), {ElementType, ElementCount},
      SourceLocation(), nullptr);

  // Create the dependent vector type: vector<element_type, element_count>.
  QualType VectorType = AST.getDependentSizedExtVectorType(
      AST.getTemplateTypeParmType(0, 0, false, ElementType),
      DeclRefExpr::Create(
          AST, NestedNameSpecifierLoc(), SourceLocation(), ElementCount, false,
          DeclarationNameInfo(ElementCount->getDeclName(), SourceLocation()),
          AST.IntTy, VK_LValue),
      SourceLocation());

  // Create the partial specialization declaration.
  QualType CanonInjectedTST =
      AST.getCanonicalType(AST.getTemplateSpecializationType(
          ElaboratedTypeKeyword::Class, TemplateName(TextureTemplate),
          {TemplateArgument(VectorType)}, {}));

  auto *PartialSpec = ClassTemplatePartialSpecializationDecl::Create(
      AST, TagDecl::TagKind::Class, HLSLNamespace, SourceLocation(),
      SourceLocation(), TemplateParams, TextureTemplate,
      {TemplateArgument(VectorType)},
      CanQualType::CreateUnsafe(CanonInjectedTST), nullptr);

  // Set the template arguments as written.
  TemplateArgument Arg(VectorType);
  TemplateArgumentLoc ArgLoc =
      S.getTrivialTemplateArgumentLoc(Arg, QualType(), SourceLocation());
  TemplateArgumentListInfo ArgsInfo =
      TemplateArgumentListInfo(SourceLocation(), SourceLocation());
  ArgsInfo.addArgument(ArgLoc);
  PartialSpec->setTemplateArgsAsWritten(
      ASTTemplateArgumentListInfo::Create(AST, ArgsInfo));

  PartialSpec->setImplicit(true);
  PartialSpec->setLexicalDeclContext(HLSLNamespace);
  PartialSpec->setHasExternalLexicalStorage();

  // Add the partial specialization to the namespace and the class template.
  HLSLNamespace->addDecl(PartialSpec);
  TextureTemplate->AddPartialSpecialization(PartialSpec, {});

  return PartialSpec;
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
// template<typename T> concept is_constant_buffer_element_compatible =
//     std::is_class_v<T> && !__is_intangible(T);
static Expr *constructConstantBufferConstraintExpr(Sema &S,
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

  TypeTraitExpr *ResExpr = TypeTraitExpr::Create(
      Context, BoolTy, NameLoc, UTT_IsConstantBufferElementCompatible,
      {TTypeSourceInfo}, NameLoc, true);

  return ResExpr;
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

enum class HLSLBufferType { Typed, Structured, Constant };

static ConceptDecl *constructBufferConceptDecl(Sema &S, NamespaceDecl *NSD,
                                               HLSLBufferType BT) {
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

  switch (BT) {
  case HLSLBufferType::Typed:
    DeclName = DeclarationName(
        &Context.Idents.get("__is_typed_resource_element_compatible"));
    ConstraintExpr = constructTypedBufferConstraintExpr(S, DeclLoc, T);
    break;
  case HLSLBufferType::Structured:
    DeclName = DeclarationName(
        &Context.Idents.get("__is_structured_resource_element_compatible"));
    ConstraintExpr = constructStructuredBufferConstraintExpr(S, DeclLoc, T);
    break;
  case HLSLBufferType::Constant:
    DeclName = DeclarationName(
        &Context.Idents.get("__is_constant_buffer_element_compatible"));
    ConstraintExpr = constructConstantBufferConstraintExpr(S, DeclLoc, T);
    break;
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
  ASTContext &AST = SemaPtr->getASTContext();
  CXXRecordDecl *Decl;
  ConceptDecl *TypedBufferConcept = constructBufferConceptDecl(
      *SemaPtr, HLSLNamespace, HLSLBufferType::Typed);
  ConceptDecl *StructuredBufferConcept = constructBufferConceptDecl(
      *SemaPtr, HLSLNamespace, HLSLBufferType::Structured);
  ConceptDecl *ConstantBufferConcept = constructBufferConceptDecl(
      *SemaPtr, HLSLNamespace, HLSLBufferType::Constant);

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "ConstantBuffer")
             .addSimpleTemplateParams({"element_type"}, ConstantBufferConcept)
             .finalizeForwardDeclaration();

  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::CBuffer, /*IsROV=*/false,
                    /*RawBuffer=*/false, /*HasCounter=*/false)
        .addConstantBufferConversionToType()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "Buffer")
             .addSimpleTemplateParams({"element_type"}, TypedBufferConcept)
             .finalizeForwardDeclaration();

  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, /*IsROV=*/false,
                    /*RawBuffer=*/false, /*HasCounter=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWBuffer")
             .addSimpleTemplateParams({"element_type"}, TypedBufferConcept)
             .finalizeForwardDeclaration();

  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/false, /*HasCounter=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RasterizerOrderedBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/true,
                    /*RawBuffer=*/false, /*HasCounter=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "StructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, /*IsROV=*/false,
                    /*RawBuffer=*/true, /*HasCounter=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWStructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true, /*HasCounter=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addIncrementCounterMethod()
        .addDecrementCounterMethod()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "AppendStructuredBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true, /*HasCounter=*/true)
        .addAppendMethod()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "ConsumeStructuredBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true, /*HasCounter=*/true)
        .addConsumeMethod()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace,
                                "RasterizerOrderedStructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/true,
                    /*RawBuffer=*/true, /*HasCounter=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addIncrementCounterMethod()
        .addDecrementCounterMethod()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "ByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, /*IsROV=*/false,
                    /*RawBuffer=*/true, /*HasCounter=*/false)
        .addByteAddressBufferLoadMethods()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/false,
                    /*RawBuffer=*/true, /*HasCounter=*/false)
        .addByteAddressBufferLoadMethods()
        .addByteAddressBufferStoreMethods()
        .addByteAddressBufferInterlockedMethods()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace,
                                "RasterizerOrderedByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, /*IsROV=*/true,
                    /*RawBuffer=*/true, /*HasCounter=*/false)
        .addByteAddressBufferInterlockedMethods()
        .addGetDimensionsMethodForBuffer()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "SamplerState")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupSamplerType(Decl, *SemaPtr).completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "SamplerComparisonState")
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupSamplerType(Decl, *SemaPtr).completeDefinition();
  });

  QualType Float4Ty = AST.getExtVectorType(AST.FloatTy, 4);
  for (const TextureTypeInfo &T : TextureTypes) {
    BuiltinTypeDeclBuilder TexBuilder(*SemaPtr, HLSLNamespace, T.Name);
    switch (T.Shape) {
    case TemplateShape::ElementType:
      TexBuilder.addSimpleTemplateParams({"element_type"}, {Float4Ty},
                                         TypedBufferConcept);
      break;
    case TemplateShape::ElementTypeAndSampleCount:
      TexBuilder.addMSTextureTemplateParams("element_type", "sample_count",
                                            TypedBufferConcept);
      break;
    }
    Decl = TexBuilder.finalizeForwardDeclaration();

    onCompletion(Decl, [this, &T](CXXRecordDecl *Decl) {
      setupTextureType(Decl, *SemaPtr, T).completeDefinition();
    });

    if (T.Shape != TemplateShape::ElementType)
      continue;

    CXXRecordDecl *PartialSpec = addVectorTexturePartialSpecialization(
        *SemaPtr, HLSLNamespace, Decl->getDescribedClassTemplate());
    onCompletion(PartialSpec, [this, &T](CXXRecordDecl *Decl) {
      setupTextureType(Decl, *SemaPtr, T).completeDefinition();
    });
  }
}

// Build a single overload of an HLSL atomic intrinsic in the hlsl namespace.
// `dest` is an address-space-qualified reference; `original_value` (when
// present) is a plain reference. The synthesized FunctionDecl aliases the
// underlying clang builtin via BuiltinAliasAttr.
static void buildAtomicOverload(Sema &S, NamespaceDecl *NS, StringRef FuncName,
                                StringRef BuiltinName, QualType ElemTy,
                                LangAS DestAS, bool ThreeArg) {
  ASTContext &AST = S.getASTContext();

  QualType DestTy =
      AST.getLValueReferenceType(AST.getAddrSpaceQualType(ElemTy, DestAS));
  QualType OrigRefTy = AST.getLValueReferenceType(ElemTy);

  SmallVector<QualType, 3> ParamTypes;
  ParamTypes.push_back(DestTy);
  ParamTypes.push_back(ElemTy);
  if (ThreeArg)
    ParamTypes.push_back(OrigRefTy);

  FunctionProtoType::ExtProtoInfo EPI;
  QualType FuncTy = AST.getFunctionType(AST.VoidTy, ParamTypes, EPI);
  auto *TSInfo = AST.getTrivialTypeSourceInfo(FuncTy, SourceLocation());

  IdentifierInfo &FuncII = AST.Idents.get(FuncName, tok::TokenKind::identifier);
  DeclarationName FuncDeclName(&FuncII);

  FunctionDecl *FD = FunctionDecl::Create(
      AST, NS, SourceLocation(), SourceLocation(), FuncDeclName, FuncTy, TSInfo,
      SC_Extern, /*UsesFPIntrin=*/false, /*isInlineSpecified=*/false,
      /*hasWrittenPrototype=*/true);

  constexpr const char *ParamNames[] = {"dest", "value", "original_value"};
  SmallVector<ParmVarDecl *, 3> ParmDecls;
  unsigned I = 0;
  for (auto [ParamType, ParamName] : llvm::zip(ParamTypes, ParamNames)) {
    IdentifierInfo &PII = AST.Idents.get(ParamName, tok::TokenKind::identifier);
    ParmVarDecl *Parm = ParmVarDecl::Create(
        AST, FD, SourceLocation(), SourceLocation(), &PII, ParamType,
        AST.getTrivialTypeSourceInfo(ParamType, SourceLocation()), SC_None,
        nullptr);
    Parm->setScopeInfo(0, I++);
    ParmDecls.push_back(Parm);
  }
  FD->setParams(ParmDecls);

  IdentifierInfo &BuiltinII =
      S.getPreprocessor().getIdentifierTable().get(BuiltinName);
  FD->addAttr(BuiltinAliasAttr::CreateImplicit(AST, &BuiltinII));
  FD->setImplicit();
  NS->addDecl(FD);
}

// Synthesize the InterlockedFunc overload set: {int, uint, int64_t, uint64_t}
// x {groupshared, device} x {2-arg, 3-arg}.
static void defineHLSLInterlockedFunc(Sema &S, NamespaceDecl *NS,
                                      StringRef FuncName,
                                      StringRef BuiltinName) {
  ASTContext &AST = S.getASTContext();
  // HLSL: int64_t == long, uint64_t == unsigned long (see hlsl_basic_types.h).
  QualType Elems[] = {AST.IntTy, AST.UnsignedIntTy, AST.LongTy,
                      AST.UnsignedLongTy};
  LangAS AddrSpaces[] = {LangAS::hlsl_groupshared, LangAS::hlsl_device};

  for (QualType ElemTy : Elems)
    for (LangAS AS : AddrSpaces)
      for (bool ThreeArg : {false, true})
        buildAtomicOverload(S, NS, FuncName, BuiltinName, ElemTy, AS, ThreeArg);
}

void HLSLExternalSemaSource::defineHLSLAtomicIntrinsics() {
  defineHLSLInterlockedFunc(*SemaPtr, HLSLNamespace, "InterlockedAdd",
                            "__builtin_hlsl_interlocked_add");
  defineHLSLInterlockedFunc(*SemaPtr, HLSLNamespace, "InterlockedMin",
                            "__builtin_hlsl_interlocked_min");
  defineHLSLInterlockedFunc(*SemaPtr, HLSLNamespace, "InterlockedOr",
                            "__builtin_hlsl_interlocked_or");
  defineHLSLInterlockedFunc(*SemaPtr, HLSLNamespace, "InterlockedXor",
                            "__builtin_hlsl_interlocked_xor");
}

void HLSLExternalSemaSource::onCompletion(CXXRecordDecl *Record,
                                          CompletionFunction Fn) {
  if (!Record->isCompleteDefinition())
    Completions.insert(std::make_pair(Record->getCanonicalDecl(), Fn));
}

void HLSLExternalSemaSource::CompleteType(TagDecl *Tag) {
  if (!isa<CXXRecordDecl>(Tag))
    return;
  auto *Record = cast<CXXRecordDecl>(Tag);
  Record = Record->getCanonicalDecl();
  auto It = Completions.find(Record);
  if (It == Completions.end())
    return;
  // Move out the callback and erase before invoking it: the callback can
  // re-enter CompleteType and mutate Completions, which invalidates It under
  // backward-shift deletion.
  CompletionFunction Fn = std::move(It->second);
  Completions.erase(It);
  Fn(Record);
}
