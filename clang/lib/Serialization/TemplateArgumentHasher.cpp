//===- TemplateArgumentHasher.cpp - Hash Template Arguments -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TemplateArgumentHasher.h"
#include "clang/AST/APValue.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/TimeProfiler.h"

using namespace clang;

namespace {

class TemplateArgumentHasher {
  // If we bail out during the process of calculating hash values for
  // template arguments for any reason. We're allowed to do it since
  // TemplateArgumentHasher are only required to give the same hash value
  // for the same template arguments, but not required to give different
  // hash value for different template arguments.
  //
  // So in the worst case, it is still a valid implementation to give all
  // inputs the same BailedOutValue as output.
  bool BailedOut = false;
  static constexpr unsigned BailedOutValue = 0x12345678;

  llvm::FoldingSetNodeID ID;

public:
  TemplateArgumentHasher() = default;

  void AddTemplateArgument(TemplateArgument TA);

  void AddInteger(unsigned V) { ID.AddInteger(V); }

  unsigned getValue() {
    if (BailedOut)
      return BailedOutValue;

    return ID.computeStableHash();
  }

  void setBailedOut() { BailedOut = true; }

  void AddType(const Type *T);
  void AddQualType(QualType T);
  void AddDecl(const Decl *D);
  void AddStructuralValue(const APValue &);
  void AddTemplateName(TemplateName Name);
  void AddDeclarationName(DeclarationName Name);
  void AddIdentifierInfo(const IdentifierInfo *II);
};

void TemplateArgumentHasher::AddTemplateArgument(TemplateArgument TA) {
  const auto Kind = TA.getKind();
  AddInteger(Kind);

  switch (Kind) {
  case TemplateArgument::Null:
    // These can occur in incomplete substitutions performed with code
    // completion (see PartialOverloading).
    break;
  case TemplateArgument::Type:
    AddQualType(TA.getAsType());
    break;
  case TemplateArgument::Declaration:
    AddDecl(TA.getAsDecl());
    break;
  case TemplateArgument::NullPtr:
    ID.AddPointer(nullptr);
    break;
  case TemplateArgument::Integral: {
    // There are integrals (e.g.: _BitInt(128)) that cannot be represented as
    // any builtin integral type, so we use the hash of APSInt instead.
    TA.getAsIntegral().Profile(ID);
    break;
  }
  case TemplateArgument::StructuralValue:
    AddQualType(TA.getStructuralValueType());
    AddStructuralValue(TA.getAsStructuralValue());
    break;
  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    AddTemplateName(TA.getAsTemplateOrTemplatePattern());
    break;
  case TemplateArgument::Expression:
    // If we meet expression in template argument, it implies
    // that the template is still dependent. It is meaningless
    // to get a stable hash for the template. Bail out simply.
    BailedOut = true;
    break;
  case TemplateArgument::Pack:
    AddInteger(TA.pack_size());
    for (auto SubTA : TA.pack_elements()) {
      AddTemplateArgument(SubTA);
    }
    break;
  }
}

void TemplateArgumentHasher::AddStructuralValue(const APValue &Value) {
  auto Kind = Value.getKind();
  AddInteger(Kind);

  // 'APValue::Profile' uses pointer values to make hash for LValue and
  // MemberPointer, but they differ from one compiler invocation to another.
  // It may be difficult to handle such cases. Bail out simply.

  if (Kind == APValue::LValue || Kind == APValue::MemberPointer) {
    BailedOut = true;
    return;
  }

  Value.Profile(ID);
}

void TemplateArgumentHasher::AddTemplateName(TemplateName Name) {
  switch (Name.getKind()) {
  case TemplateName::Template:
    AddDecl(Name.getAsTemplateDecl());
    break;
  case TemplateName::QualifiedTemplate: {
    QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName();
    AddTemplateName(QTN->getUnderlyingTemplate());
    break;
  }
  case TemplateName::OverloadedTemplate:
  case TemplateName::AssumedTemplate:
  case TemplateName::DependentTemplate:
  case TemplateName::SubstTemplateTemplateParm:
  case TemplateName::SubstTemplateTemplateParmPack:
    BailedOut = true;
    break;
  case TemplateName::UsingTemplate: {
    UsingShadowDecl *USD = Name.getAsUsingShadowDecl();
    if (USD)
      AddDecl(USD->getTargetDecl());
    else
      BailedOut = true;
    break;
  }
  case TemplateName::DeducedTemplate:
    AddTemplateName(Name.getAsDeducedTemplateName()->getUnderlying());
    break;
  }
}

void TemplateArgumentHasher::AddIdentifierInfo(const IdentifierInfo *II) {
  assert(II && "Expecting non-null pointer.");
  ID.AddString(II->getName());
}

void TemplateArgumentHasher::AddDeclarationName(DeclarationName Name) {
  if (Name.isEmpty())
    return;

  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    AddIdentifierInfo(Name.getAsIdentifierInfo());
    break;
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    BailedOut = true;
    break;
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
    AddQualType(Name.getCXXNameType());
    break;
  case DeclarationName::CXXOperatorName:
    AddInteger(Name.getCXXOverloadedOperator());
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
    if (auto *Template = Name.getCXXDeductionGuideTemplate())
      AddDecl(Template);
  }
  }
}

void TemplateArgumentHasher::AddDecl(const Decl *D) {
  const NamedDecl *ND = dyn_cast<NamedDecl>(D);
  if (!ND) {
    BailedOut = true;
    return;
  }

  AddDeclarationName(ND->getDeclName());
}

void TemplateArgumentHasher::AddQualType(QualType T) {
  if (T.isNull()) {
    BailedOut = true;
    return;
  }
  SplitQualType split = T.split();
  AddInteger(split.Quals.getAsOpaqueValue());
  AddType(split.Ty);
}

// Process a Type pointer.  Add* methods call back into TemplateArgumentHasher
// while Visit* methods process the relevant parts of the Type.
// Any unhandled type will make the hash computation bail out.
class TypeVisitorHelper : public TypeVisitor<TypeVisitorHelper> {
  typedef TypeVisitor<TypeVisitorHelper> Inherited;
  llvm::FoldingSetNodeID &ID;
  TemplateArgumentHasher &Hash;

public:
  TypeVisitorHelper(llvm::FoldingSetNodeID &ID, TemplateArgumentHasher &Hash)
      : ID(ID), Hash(Hash) {}

  void AddDecl(const Decl *D) {
    if (D)
      Hash.AddDecl(D);
    else
      Hash.AddInteger(0);
  }

  void AddQualType(QualType T) { Hash.AddQualType(T); }

  void AddType(const Type *T) {
    if (T)
      Hash.AddType(T);
    else
      Hash.AddInteger(0);
  }

  void VisitQualifiers(Qualifiers Quals) {
    Hash.AddInteger(Quals.getAsOpaqueValue());
  }

  void Visit(const Type *T) { Inherited::Visit(T); }

  // Unhandled types. Bail out simply.
  void VisitType(const Type *T) { Hash.setBailedOut(); }

  void VisitAdjustedType(const AdjustedType *T) {
    AddQualType(T->getOriginalType());
  }

  void VisitDecayedType(const DecayedType *T) {
    // getDecayedType and getPointeeType are derived from getAdjustedType
    // and don't need to be separately processed.
    VisitAdjustedType(T);
  }

  void VisitArrayType(const ArrayType *T) {
    AddQualType(T->getElementType());
    Hash.AddInteger(llvm::to_underlying(T->getSizeModifier()));
    VisitQualifiers(T->getIndexTypeQualifiers());
  }
  void VisitConstantArrayType(const ConstantArrayType *T) {
    T->getSize().Profile(ID);
    VisitArrayType(T);
  }

  void VisitAttributedType(const AttributedType *T) {
    Hash.AddInteger(T->getAttrKind());
    AddQualType(T->getModifiedType());
  }

  void VisitBuiltinType(const BuiltinType *T) { Hash.AddInteger(T->getKind()); }

  void VisitComplexType(const ComplexType *T) {
    AddQualType(T->getElementType());
  }

  void VisitDecltypeType(const DecltypeType *T) {
    AddQualType(T->getUnderlyingType());
  }

  void VisitDeducedType(const DeducedType *T) {
    AddQualType(T->getDeducedType());
  }

  void VisitAutoType(const AutoType *T) { VisitDeducedType(T); }

  void VisitDeducedTemplateSpecializationType(
      const DeducedTemplateSpecializationType *T) {
    Hash.AddTemplateName(T->getTemplateName());
    VisitDeducedType(T);
  }

  void VisitFunctionType(const FunctionType *T) {
    AddQualType(T->getReturnType());
    T->getExtInfo().Profile(ID);
    Hash.AddInteger(T->isConst());
    Hash.AddInteger(T->isVolatile());
    Hash.AddInteger(T->isRestrict());
  }

  void VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
    VisitFunctionType(T);
  }

  void VisitFunctionProtoType(const FunctionProtoType *T) {
    Hash.AddInteger(T->getNumParams());
    for (auto ParamType : T->getParamTypes())
      AddQualType(ParamType);

    VisitFunctionType(T);
  }

  void VisitMemberPointerType(const MemberPointerType *T) {
    AddQualType(T->getPointeeType());
    AddType(T->getQualifier().getAsType());
    if (auto *RD = T->getMostRecentCXXRecordDecl())
      AddDecl(RD->getCanonicalDecl());
  }

  void VisitPackExpansionType(const PackExpansionType *T) {
    AddQualType(T->getPattern());
  }

  void VisitParenType(const ParenType *T) { AddQualType(T->getInnerType()); }

  void VisitPointerType(const PointerType *T) {
    AddQualType(T->getPointeeType());
  }

  void VisitReferenceType(const ReferenceType *T) {
    AddQualType(T->getPointeeTypeAsWritten());
  }

  void VisitLValueReferenceType(const LValueReferenceType *T) {
    VisitReferenceType(T);
  }

  void VisitRValueReferenceType(const RValueReferenceType *T) {
    VisitReferenceType(T);
  }

  void
  VisitSubstTemplateTypeParmPackType(const SubstTemplateTypeParmPackType *T) {
    AddDecl(T->getAssociatedDecl());
    Hash.AddTemplateArgument(T->getArgumentPack());
  }

  void VisitSubstTemplateTypeParmType(const SubstTemplateTypeParmType *T) {
    AddDecl(T->getAssociatedDecl());
    AddQualType(T->getReplacementType());
  }

  void VisitTagType(const TagType *T) { AddDecl(T->getOriginalDecl()); }

  void VisitRecordType(const RecordType *T) { VisitTagType(T); }
  void VisitEnumType(const EnumType *T) { VisitTagType(T); }

  void VisitTemplateSpecializationType(const TemplateSpecializationType *T) {
    Hash.AddInteger(T->template_arguments().size());
    for (const auto &TA : T->template_arguments()) {
      Hash.AddTemplateArgument(TA);
    }
    Hash.AddTemplateName(T->getTemplateName());
  }

  void VisitTemplateTypeParmType(const TemplateTypeParmType *T) {
    Hash.AddInteger(T->getDepth());
    Hash.AddInteger(T->getIndex());
    Hash.AddInteger(T->isParameterPack());
  }

  void VisitTypedefType(const TypedefType *T) { AddDecl(T->getDecl()); }

  void VisitUnaryTransformType(const UnaryTransformType *T) {
    AddQualType(T->getUnderlyingType());
    AddQualType(T->getBaseType());
  }

  void VisitVectorType(const VectorType *T) {
    AddQualType(T->getElementType());
    Hash.AddInteger(T->getNumElements());
    Hash.AddInteger(llvm::to_underlying(T->getVectorKind()));
  }

  void VisitExtVectorType(const ExtVectorType *T) { VisitVectorType(T); }
};

void TemplateArgumentHasher::AddType(const Type *T) {
  assert(T && "Expecting non-null pointer.");
  TypeVisitorHelper(ID, *this).Visit(T);
}

} // namespace

unsigned clang::serialization::StableHashForTemplateArguments(
    llvm::ArrayRef<TemplateArgument> Args) {
  llvm::TimeTraceScope TimeScope("Stable Hash for Template Arguments");
  TemplateArgumentHasher Hasher;
  Hasher.AddInteger(Args.size());
  for (TemplateArgument Arg : Args)
    Hasher.AddTemplateArgument(Arg);
  return Hasher.getValue();
}
