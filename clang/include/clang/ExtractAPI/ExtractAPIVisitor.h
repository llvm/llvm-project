//===- ExtractAPI/ExtractAPIVisitor.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ExtractAPVisitor AST visitation interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_EXTRACT_API_VISITOR_H
#define LLVM_CLANG_EXTRACTAPI_EXTRACT_API_VISITOR_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/Specifiers.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "llvm/ADT/FunctionExtras.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/TypedefUnderlyingTypeResolver.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/StringRef.h"
#include <type_traits>

namespace clang {
namespace extractapi {
namespace impl {

template <typename Derived>
class ExtractAPIVisitorBase : public RecursiveASTVisitor<Derived> {
protected:
  ExtractAPIVisitorBase(ASTContext &Context, APISet &API)
      : Context(Context), API(API) {}

public:
  const APISet &getAPI() const { return API; }

  bool VisitVarDecl(const VarDecl *Decl);

  bool VisitFunctionDecl(const FunctionDecl *Decl);

  bool VisitEnumDecl(const EnumDecl *Decl);

  bool WalkUpFromFunctionDecl(const FunctionDecl *Decl);

  bool WalkUpFromRecordDecl(const RecordDecl *Decl);

  bool WalkUpFromCXXRecordDecl(const CXXRecordDecl *Decl);

  bool WalkUpFromCXXMethodDecl(const CXXMethodDecl *Decl);

  bool WalkUpFromClassTemplateSpecializationDecl(
      const ClassTemplateSpecializationDecl *Decl);

  bool WalkUpFromClassTemplatePartialSpecializationDecl(
      const ClassTemplatePartialSpecializationDecl *Decl);

  bool WalkUpFromVarTemplateDecl(const VarTemplateDecl *Decl);

  bool WalkUpFromVarTemplateSpecializationDecl(
      const VarTemplateSpecializationDecl *Decl);

  bool WalkUpFromVarTemplatePartialSpecializationDecl(
      const VarTemplatePartialSpecializationDecl *Decl);

  bool WalkUpFromFunctionTemplateDecl(const FunctionTemplateDecl *Decl);

  bool WalkUpFromNamespaceDecl(const NamespaceDecl *Decl);

  bool VisitNamespaceDecl(const NamespaceDecl *Decl);

  bool VisitRecordDecl(const RecordDecl *Decl);

  bool VisitCXXRecordDecl(const CXXRecordDecl *Decl);

  bool VisitCXXMethodDecl(const CXXMethodDecl *Decl);

  bool VisitFieldDecl(const FieldDecl *Decl);

  bool VisitCXXConversionDecl(const CXXConversionDecl *Decl);

  bool VisitCXXConstructorDecl(const CXXConstructorDecl *Decl);

  bool VisitCXXDestructorDecl(const CXXDestructorDecl *Decl);

  bool VisitConceptDecl(const ConceptDecl *Decl);

  bool VisitClassTemplateSpecializationDecl(
      const ClassTemplateSpecializationDecl *Decl);

  bool VisitClassTemplatePartialSpecializationDecl(
      const ClassTemplatePartialSpecializationDecl *Decl);

  bool VisitVarTemplateDecl(const VarTemplateDecl *Decl);

  bool
  VisitVarTemplateSpecializationDecl(const VarTemplateSpecializationDecl *Decl);

  bool VisitVarTemplatePartialSpecializationDecl(
      const VarTemplatePartialSpecializationDecl *Decl);

  bool VisitFunctionTemplateDecl(const FunctionTemplateDecl *Decl);

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *Decl);

  bool VisitObjCProtocolDecl(const ObjCProtocolDecl *Decl);

  bool VisitTypedefNameDecl(const TypedefNameDecl *Decl);

  bool VisitObjCCategoryDecl(const ObjCCategoryDecl *Decl);

  bool shouldDeclBeIncluded(const Decl *Decl) const;

  const RawComment *fetchRawCommentForDecl(const Decl *Decl) const;

protected:
  /// Collect API information for the enum constants and associate with the
  /// parent enum.
  void recordEnumConstants(EnumRecord *EnumRecord,
                           const EnumDecl::enumerator_range Constants);

  /// Collect API information for the record fields and associate with the
  /// parent struct.
  void recordRecordFields(RecordRecord *RecordRecord,
                          APIRecord::RecordKind FieldKind,
                          const RecordDecl::field_range Fields);

  /// Collect API information for the Objective-C methods and associate with the
  /// parent container.
  void recordObjCMethods(ObjCContainerRecord *Container,
                         const ObjCContainerDecl::method_range Methods);

  void recordObjCProperties(ObjCContainerRecord *Container,
                            const ObjCContainerDecl::prop_range Properties);

  void recordObjCInstanceVariables(
      ObjCContainerRecord *Container,
      const llvm::iterator_range<
          DeclContext::specific_decl_iterator<ObjCIvarDecl>>
          Ivars);

  void recordObjCProtocols(ObjCContainerRecord *Container,
                           ObjCInterfaceDecl::protocol_range Protocols);

  ASTContext &Context;
  APISet &API;

  StringRef getTypedefName(const TagDecl *Decl) {
    if (const auto *TypedefDecl = Decl->getTypedefNameForAnonDecl())
      return TypedefDecl->getName();

    return {};
  }

  bool isInSystemHeader(const Decl *D) {
    return Context.getSourceManager().isInSystemHeader(D->getLocation());
  }

private:
  Derived &getDerivedExtractAPIVisitor() {
    return *static_cast<Derived *>(this);
  }

  SmallVector<SymbolReference> getBases(const CXXRecordDecl *Decl) {
    // FIXME: store AccessSpecifier given by inheritance
    SmallVector<SymbolReference> Bases;
    for (const auto &BaseSpecifier : Decl->bases()) {
      // skip classes not inherited as public
      if (BaseSpecifier.getAccessSpecifier() != AccessSpecifier::AS_public)
        continue;
      SymbolReference BaseClass;
      if (BaseSpecifier.getType().getTypePtr()->isTemplateTypeParmType()) {
        BaseClass.Name = API.copyString(BaseSpecifier.getType().getAsString());
        BaseClass.USR = API.recordUSR(
            BaseSpecifier.getType()->getAs<TemplateTypeParmType>()->getDecl());
      } else {
        CXXRecordDecl *BaseClassDecl =
            BaseSpecifier.getType().getTypePtr()->getAsCXXRecordDecl();
        BaseClass.Name = BaseClassDecl->getName();
        BaseClass.USR = API.recordUSR(BaseClassDecl);
      }
      Bases.emplace_back(BaseClass);
    }
    return Bases;
  }

  APIRecord *determineParentRecord(const DeclContext *Context) {
    SmallString<128> ParentUSR;
    if (Context->getDeclKind() == Decl::TranslationUnit)
      return nullptr;

    index::generateUSRForDecl(dyn_cast<Decl>(Context), ParentUSR);

    APIRecord *Parent = API.findRecordForUSR(ParentUSR);
    return Parent;
  }
};

template <typename T>
static void modifyRecords(const T &Records, const StringRef &Name) {
  for (const auto &Record : Records) {
    if (Name == Record.second.get()->Name) {
      auto &DeclFragment = Record.second->Declaration;
      DeclFragment.insert(DeclFragment.begin(), " ",
                          DeclarationFragments::FragmentKind::Text);
      DeclFragment.insert(DeclFragment.begin(), "typedef",
                          DeclarationFragments::FragmentKind::Keyword, "",
                          nullptr);
      DeclFragment.insert(--DeclFragment.end(), " { ... } ",
                          DeclarationFragments::FragmentKind::Text);
      DeclFragment.insert(--DeclFragment.end(), Name,
                          DeclarationFragments::FragmentKind::Identifier);
      break;
    }
  }
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitVarDecl(const VarDecl *Decl) {
  // skip function parameters.
  if (isa<ParmVarDecl>(Decl))
    return true;

  // Skip non-global variables in records (struct/union/class) but not static
  // members.
  if (Decl->getDeclContext()->isRecord() && !Decl->isStaticDataMember())
    return true;

  // Skip local variables inside function or method.
  if (!Decl->isDefinedOutsideFunctionOrMethod())
    return true;

  // If this is a template but not specialization or instantiation, skip.
  if (Decl->getASTContext().getTemplateOrSpecializationInfo(Decl) &&
      Decl->getTemplateSpecializationKind() == TSK_Undeclared)
    return true;

  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the variable.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForVar(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  if (Decl->isStaticDataMember()) {
    SymbolReference Context;
    // getDeclContext() should return a RecordDecl since we
    // are currently handling a static data member.
    auto *Record = cast<RecordDecl>(Decl->getDeclContext());
    Context.Name = Record->getName();
    Context.USR = API.recordUSR(Record);
    auto Access = DeclarationFragmentsBuilder::getAccessControl(Decl);
    API.addStaticField(Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl),
                       Linkage, Comment, Declaration, SubHeading, Context,
                       Access, isInSystemHeader(Decl));
  } else
    // Add the global variable record to the API set.
    API.addGlobalVar(Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl),
                     Linkage, Comment, Declaration, SubHeading,
                     isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitFunctionDecl(
    const FunctionDecl *Decl) {
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Decl)) {
    // Skip member function in class templates.
    if (Method->getParent()->getDescribedClassTemplate() != nullptr)
      return true;

    // Skip methods in records.
    for (const auto &P : Context.getParents(*Method)) {
      if (P.template get<CXXRecordDecl>())
        return true;
    }

    // Skip ConstructorDecl and DestructorDecl.
    if (isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method))
      return true;
  }

  // Skip templated functions.
  switch (Decl->getTemplatedKind()) {
  case FunctionDecl::TK_NonTemplate:
  case FunctionDecl::TK_DependentNonTemplate:
  case FunctionDecl::TK_FunctionTemplateSpecialization:
    break;
  case FunctionDecl::TK_FunctionTemplate:
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization:
  case FunctionDecl::TK_MemberSpecialization:
    return true;
  }

  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments, sub-heading, and signature of the function.
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  FunctionSignature Signature =
      DeclarationFragmentsBuilder::getFunctionSignature(Decl);
  if (Decl->getTemplateSpecializationInfo())
    API.addGlobalFunctionTemplateSpecialization(
        Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Linkage,
        Comment,
        DeclarationFragmentsBuilder::
            getFragmentsForFunctionTemplateSpecialization(Decl),
        SubHeading, Signature, isInSystemHeader(Decl));
  else
    // Add the function record to the API set.
    API.addGlobalFunction(
        Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Linkage,
        Comment, DeclarationFragmentsBuilder::getFragmentsForFunction(Decl),
        SubHeading, Signature, isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitEnumDecl(const EnumDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  SmallString<128> QualifiedNameBuffer;
  // Collect symbol information.
  StringRef Name = Decl->getName();
  if (Name.empty())
    Name = getTypedefName(Decl);
  if (Name.empty()) {
    llvm::raw_svector_ostream OS(QualifiedNameBuffer);
    Decl->printQualifiedName(OS);
    Name = QualifiedNameBuffer.str();
  }

  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the enum.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForEnum(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  EnumRecord *EnumRecord = API.addEnum(
      API.copyString(Name), USR, Loc, AvailabilityInfo::createFromDecl(Decl),
      Comment, Declaration, SubHeading, isInSystemHeader(Decl));

  // Now collect information about the enumerators in this enum.
  getDerivedExtractAPIVisitor().recordEnumConstants(EnumRecord,
                                                    Decl->enumerators());

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromFunctionDecl(
    const FunctionDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitFunctionDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromRecordDecl(
    const RecordDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitRecordDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromCXXRecordDecl(
    const CXXRecordDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitCXXRecordDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromCXXMethodDecl(
    const CXXMethodDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitCXXMethodDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromClassTemplateSpecializationDecl(
    const ClassTemplateSpecializationDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitClassTemplateSpecializationDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::
    WalkUpFromClassTemplatePartialSpecializationDecl(
        const ClassTemplatePartialSpecializationDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitClassTemplatePartialSpecializationDecl(
      Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromVarTemplateDecl(
    const VarTemplateDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitVarTemplateDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromVarTemplateSpecializationDecl(
    const VarTemplateSpecializationDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitVarTemplateSpecializationDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::
    WalkUpFromVarTemplatePartialSpecializationDecl(
        const VarTemplatePartialSpecializationDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitVarTemplatePartialSpecializationDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromFunctionTemplateDecl(
    const FunctionTemplateDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitFunctionTemplateDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::WalkUpFromNamespaceDecl(
    const NamespaceDecl *Decl) {
  getDerivedExtractAPIVisitor().VisitNamespaceDecl(Decl);
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitNamespaceDecl(
    const NamespaceDecl *Decl) {

  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;
  if (Decl->isAnonymousNamespace())
    return true;
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the struct.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForNamespace(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  APIRecord *Parent = determineParentRecord(Decl->getDeclContext());
  API.addNamespace(Parent, Name, USR, Loc,
                   AvailabilityInfo::createFromDecl(Decl), Linkage, Comment,
                   Declaration, SubHeading, isInSystemHeader(Decl));

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitRecordDecl(const RecordDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;
  // Collect symbol information.
  StringRef Name = Decl->getName();
  if (Name.empty())
    Name = getTypedefName(Decl);
  if (Name.empty())
    return true;

  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the struct.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForRecordDecl(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  auto RecordKind = APIRecord::RK_Struct;
  auto FieldRecordKind = APIRecord::RK_StructField;

  if (Decl->isUnion()) {
    RecordKind = APIRecord::RK_Union;
    FieldRecordKind = APIRecord::RK_UnionField;
  }

  RecordRecord *RecordRecord = API.addRecord(
      Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Comment,
      Declaration, SubHeading, RecordKind, isInSystemHeader(Decl));

  // Now collect information about the fields in this struct.
  getDerivedExtractAPIVisitor().recordRecordFields(
      RecordRecord, FieldRecordKind, Decl->fields());

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitCXXRecordDecl(
    const CXXRecordDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl) ||
      Decl->isImplicit())
    return true;

  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForCXXClass(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  APIRecord::RecordKind Kind;
  if (Decl->isUnion())
    Kind = APIRecord::RecordKind::RK_Union;
  else if (Decl->isStruct())
    Kind = APIRecord::RecordKind::RK_Struct;
  else
    Kind = APIRecord::RecordKind::RK_CXXClass;
  auto Access = DeclarationFragmentsBuilder::getAccessControl(Decl);

  APIRecord *Parent = determineParentRecord(Decl->getDeclContext());
  CXXClassRecord *CXXClassRecord;
  if (Decl->getDescribedClassTemplate()) {
    // Inject template fragments before class fragments.
    Declaration.insert(
        Declaration.begin(),
        DeclarationFragmentsBuilder::getFragmentsForRedeclarableTemplate(
            Decl->getDescribedClassTemplate()));
    CXXClassRecord = API.addClassTemplate(
        Parent, Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Comment,
        Declaration, SubHeading, Template(Decl->getDescribedClassTemplate()),
        Access, isInSystemHeader(Decl));
  } else
    CXXClassRecord = API.addCXXClass(
        Parent, Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Comment,
        Declaration, SubHeading, Kind, Access, isInSystemHeader(Decl));

  CXXClassRecord->Bases = getBases(Decl);

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitCXXMethodDecl(
    const CXXMethodDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl) ||
      Decl->isImplicit())
    return true;

  if (isa<CXXConversionDecl>(Decl))
    return true;
  if (isa<CXXConstructorDecl>(Decl) || isa<CXXDestructorDecl>(Decl))
    return true;

  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  auto Access = DeclarationFragmentsBuilder::getAccessControl(Decl);
  auto Signature = DeclarationFragmentsBuilder::getFunctionSignature(Decl);

  SmallString<128> ParentUSR;
  index::generateUSRForDecl(dyn_cast<CXXRecordDecl>(Decl->getDeclContext()),
                            ParentUSR);
  auto *Parent = API.findRecordForUSR(ParentUSR);
  if (Decl->isTemplated()) {
    FunctionTemplateDecl *TemplateDecl = Decl->getDescribedFunctionTemplate();
    API.addCXXMethodTemplate(
        API.findRecordForUSR(ParentUSR), Decl->getName(), USR, Loc,
        AvailabilityInfo::createFromDecl(Decl), Comment,
        DeclarationFragmentsBuilder::getFragmentsForFunctionTemplate(
            TemplateDecl),
        SubHeading, DeclarationFragmentsBuilder::getFunctionSignature(Decl),
        DeclarationFragmentsBuilder::getAccessControl(TemplateDecl),
        Template(TemplateDecl), isInSystemHeader(Decl));
  } else if (Decl->getTemplateSpecializationInfo())
    API.addCXXMethodTemplateSpec(
        Parent, Decl->getName(), USR, Loc,
        AvailabilityInfo::createFromDecl(Decl), Comment,
        DeclarationFragmentsBuilder::
            getFragmentsForFunctionTemplateSpecialization(Decl),
        SubHeading, Signature, Access, isInSystemHeader(Decl));
  else if (Decl->isOverloadedOperator())
    API.addCXXInstanceMethod(
        Parent, API.copyString(Decl->getNameAsString()), USR, Loc,
        AvailabilityInfo::createFromDecl(Decl), Comment,
        DeclarationFragmentsBuilder::getFragmentsForOverloadedOperator(Decl),
        SubHeading, Signature, Access, isInSystemHeader(Decl));
  else if (Decl->isStatic())
    API.addCXXStaticMethod(
        Parent, Decl->getName(), USR, Loc,
        AvailabilityInfo::createFromDecl(Decl), Comment,
        DeclarationFragmentsBuilder::getFragmentsForCXXMethod(Decl), SubHeading,
        Signature, Access, isInSystemHeader(Decl));
  else
    API.addCXXInstanceMethod(
        Parent, Decl->getName(), USR, Loc,
        AvailabilityInfo::createFromDecl(Decl), Comment,
        DeclarationFragmentsBuilder::getFragmentsForCXXMethod(Decl), SubHeading,
        Signature, Access, isInSystemHeader(Decl));

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitCXXConstructorDecl(
    const CXXConstructorDecl *Decl) {

  StringRef Name = API.copyString(Decl->getNameAsString());
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments, sub-heading, and signature for the method.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForSpecialCXXMethod(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  FunctionSignature Signature =
      DeclarationFragmentsBuilder::getFunctionSignature(Decl);
  AccessControl Access = DeclarationFragmentsBuilder::getAccessControl(Decl);
  SmallString<128> ParentUSR;
  index::generateUSRForDecl(dyn_cast<CXXRecordDecl>(Decl->getDeclContext()),
                            ParentUSR);
  API.addCXXInstanceMethod(API.findRecordForUSR(ParentUSR), Name, USR, Loc,
                           AvailabilityInfo::createFromDecl(Decl), Comment,
                           Declaration, SubHeading, Signature, Access,
                           isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitCXXDestructorDecl(
    const CXXDestructorDecl *Decl) {

  StringRef Name = API.copyString(Decl->getNameAsString());
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments, sub-heading, and signature for the method.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForSpecialCXXMethod(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  FunctionSignature Signature =
      DeclarationFragmentsBuilder::getFunctionSignature(Decl);
  AccessControl Access = DeclarationFragmentsBuilder::getAccessControl(Decl);
  SmallString<128> ParentUSR;
  index::generateUSRForDecl(dyn_cast<CXXRecordDecl>(Decl->getDeclContext()),
                            ParentUSR);
  API.addCXXInstanceMethod(API.findRecordForUSR(ParentUSR), Name, USR, Loc,
                           AvailabilityInfo::createFromDecl(Decl), Comment,
                           Declaration, SubHeading, Signature, Access,
                           isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitConceptDecl(const ConceptDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForConcept(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  API.addConcept(Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl),
                 Comment, Declaration, SubHeading, Template(Decl),
                 isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitClassTemplateSpecializationDecl(
    const ClassTemplateSpecializationDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForClassTemplateSpecialization(
          Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  APIRecord *Parent = determineParentRecord(Decl->getDeclContext());
  auto *ClassTemplateSpecializationRecord = API.addClassTemplateSpecialization(
      Parent, Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Comment,
      Declaration, SubHeading,
      DeclarationFragmentsBuilder::getAccessControl(Decl),
      isInSystemHeader(Decl));

  ClassTemplateSpecializationRecord->Bases = getBases(Decl);

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::
    VisitClassTemplatePartialSpecializationDecl(
        const ClassTemplatePartialSpecializationDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());
  DeclarationFragments Declaration = DeclarationFragmentsBuilder::
      getFragmentsForClassTemplatePartialSpecialization(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  APIRecord *Parent = determineParentRecord(Decl->getDeclContext());
  auto *ClassTemplatePartialSpecRecord =
      API.addClassTemplatePartialSpecialization(
          Parent, Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl),
          Comment, Declaration, SubHeading, Template(Decl),
          DeclarationFragmentsBuilder::getAccessControl(Decl),
          isInSystemHeader(Decl));

  ClassTemplatePartialSpecRecord->Bases = getBases(Decl);

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitVarTemplateDecl(
    const VarTemplateDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the variable.
  DeclarationFragments Declaration;
  Declaration
      .append(DeclarationFragmentsBuilder::getFragmentsForRedeclarableTemplate(
          Decl))
      .append(DeclarationFragmentsBuilder::getFragmentsForVarTemplate(
          Decl->getTemplatedDecl()));
  // Inject template fragments before var fragments.
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  SmallString<128> ParentUSR;
  index::generateUSRForDecl(dyn_cast<CXXRecordDecl>(Decl->getDeclContext()),
                            ParentUSR);
  if (Decl->getDeclContext()->getDeclKind() == Decl::CXXRecord)
    API.addCXXFieldTemplate(API.findRecordForUSR(ParentUSR), Name, USR, Loc,
                            AvailabilityInfo::createFromDecl(Decl), Comment,
                            Declaration, SubHeading,
                            DeclarationFragmentsBuilder::getAccessControl(Decl),
                            Template(Decl), isInSystemHeader(Decl));
  else
    API.addGlobalVariableTemplate(Name, USR, Loc,
                                  AvailabilityInfo::createFromDecl(Decl),
                                  Linkage, Comment, Declaration, SubHeading,
                                  Template(Decl), isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitVarTemplateSpecializationDecl(
    const VarTemplateSpecializationDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the variable.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForVarTemplateSpecialization(
          Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  API.addGlobalVariableTemplateSpecialization(
      Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Linkage, Comment,
      Declaration, SubHeading, isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitVarTemplatePartialSpecializationDecl(
    const VarTemplatePartialSpecializationDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the variable.
  DeclarationFragments Declaration = DeclarationFragmentsBuilder::
      getFragmentsForVarTemplatePartialSpecialization(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  API.addGlobalVariableTemplatePartialSpecialization(
      Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Linkage, Comment,
      Declaration, SubHeading, Template(Decl), isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitFunctionTemplateDecl(
    const FunctionTemplateDecl *Decl) {
  if (isa<CXXMethodDecl>(Decl->getTemplatedDecl()))
    return true;
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  FunctionSignature Signature =
      DeclarationFragmentsBuilder::getFunctionSignature(
          Decl->getTemplatedDecl());
  API.addGlobalFunctionTemplate(
      Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Linkage, Comment,
      DeclarationFragmentsBuilder::getFragmentsForFunctionTemplate(Decl),
      SubHeading, Signature, Template(Decl), isInSystemHeader(Decl));

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitObjCInterfaceDecl(
    const ObjCInterfaceDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the interface.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForObjCInterface(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  // Collect super class information.
  SymbolReference SuperClass;
  if (const auto *SuperClassDecl = Decl->getSuperClass()) {
    SuperClass.Name = SuperClassDecl->getObjCRuntimeNameAsString();
    SuperClass.USR = API.recordUSR(SuperClassDecl);
  }

  ObjCInterfaceRecord *ObjCInterfaceRecord = API.addObjCInterface(
      Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Linkage, Comment,
      Declaration, SubHeading, SuperClass, isInSystemHeader(Decl));

  // Record all methods (selectors). This doesn't include automatically
  // synthesized property methods.
  getDerivedExtractAPIVisitor().recordObjCMethods(ObjCInterfaceRecord,
                                                  Decl->methods());
  getDerivedExtractAPIVisitor().recordObjCProperties(ObjCInterfaceRecord,
                                                     Decl->properties());
  getDerivedExtractAPIVisitor().recordObjCInstanceVariables(ObjCInterfaceRecord,
                                                            Decl->ivars());
  getDerivedExtractAPIVisitor().recordObjCProtocols(ObjCInterfaceRecord,
                                                    Decl->protocols());

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitObjCProtocolDecl(
    const ObjCProtocolDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the protocol.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForObjCProtocol(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  ObjCProtocolRecord *ObjCProtocolRecord = API.addObjCProtocol(
      Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Comment,
      Declaration, SubHeading, isInSystemHeader(Decl));

  getDerivedExtractAPIVisitor().recordObjCMethods(ObjCProtocolRecord,
                                                  Decl->methods());
  getDerivedExtractAPIVisitor().recordObjCProperties(ObjCProtocolRecord,
                                                     Decl->properties());
  getDerivedExtractAPIVisitor().recordObjCProtocols(ObjCProtocolRecord,
                                                    Decl->protocols());

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitTypedefNameDecl(
    const TypedefNameDecl *Decl) {
  // Skip ObjC Type Parameter for now.
  if (isa<ObjCTypeParamDecl>(Decl))
    return true;

  if (!Decl->isDefinedOutsideFunctionOrMethod())
    return true;

  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  // Add the notion of typedef for tag type (struct or enum) of the same name.
  if (const ElaboratedType *ET =
          dyn_cast<ElaboratedType>(Decl->getUnderlyingType())) {
    if (const TagType *TagTy = dyn_cast<TagType>(ET->desugar())) {
      if (Decl->getName() == TagTy->getDecl()->getName()) {
        if (isa<RecordDecl>(TagTy->getDecl())) {
          modifyRecords(API.getRecords(), Decl->getName());
        }
        if (TagTy->getDecl()->isEnum()) {
          modifyRecords(API.getEnums(), Decl->getName());
        }
      }
    }
  }

  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  QualType Type = Decl->getUnderlyingType();
  SymbolReference SymRef =
      TypedefUnderlyingTypeResolver(Context).getSymbolReferenceForType(Type,
                                                                       API);

  API.addTypedef(Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl),
                 Comment,
                 DeclarationFragmentsBuilder::getFragmentsForTypedef(Decl),
                 DeclarationFragmentsBuilder::getSubHeading(Decl), SymRef,
                 isInSystemHeader(Decl));

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitObjCCategoryDecl(
    const ObjCCategoryDecl *Decl) {
  if (!getDerivedExtractAPIVisitor().shouldDeclBeIncluded(Decl))
    return true;

  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());
  // Build declaration fragments and sub-heading for the category.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForObjCCategory(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  const ObjCInterfaceDecl *InterfaceDecl = Decl->getClassInterface();
  SymbolReference Interface(InterfaceDecl->getName(),
                            API.recordUSR(InterfaceDecl));

  bool IsFromExternalModule = true;
  for (const auto &Interface : API.getObjCInterfaces()) {
    if (InterfaceDecl->getName() == Interface.second.get()->Name) {
      IsFromExternalModule = false;
      break;
    }
  }

  ObjCCategoryRecord *ObjCCategoryRecord = API.addObjCCategory(
      Name, USR, Loc, AvailabilityInfo::createFromDecl(Decl), Comment,
      Declaration, SubHeading, Interface, isInSystemHeader(Decl),
      IsFromExternalModule);

  getDerivedExtractAPIVisitor().recordObjCMethods(ObjCCategoryRecord,
                                                  Decl->methods());
  getDerivedExtractAPIVisitor().recordObjCProperties(ObjCCategoryRecord,
                                                     Decl->properties());
  getDerivedExtractAPIVisitor().recordObjCInstanceVariables(ObjCCategoryRecord,
                                                            Decl->ivars());
  getDerivedExtractAPIVisitor().recordObjCProtocols(ObjCCategoryRecord,
                                                    Decl->protocols());

  return true;
}

/// Collect API information for the enum constants and associate with the
/// parent enum.
template <typename Derived>
void ExtractAPIVisitorBase<Derived>::recordEnumConstants(
    EnumRecord *EnumRecord, const EnumDecl::enumerator_range Constants) {
  for (const auto *Constant : Constants) {
    // Collect symbol information.
    StringRef Name = Constant->getName();
    StringRef USR = API.recordUSR(Constant);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Constant->getLocation());
    DocComment Comment;
    if (auto *RawComment =
            getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Constant))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the enum constant.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForEnumConstant(Constant);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Constant);

    API.addEnumConstant(EnumRecord, Name, USR, Loc,
                        AvailabilityInfo::createFromDecl(Constant), Comment,
                        Declaration, SubHeading, isInSystemHeader(Constant));
  }
}

/// Collect API information for the struct fields and associate with the
/// parent struct.
template <typename Derived>
void ExtractAPIVisitorBase<Derived>::recordRecordFields(
    RecordRecord *RecordRecord, APIRecord::RecordKind FieldKind,
    const RecordDecl::field_range Fields) {
  for (const auto *Field : Fields) {
    // Collect symbol information.
    StringRef Name = Field->getName();
    StringRef USR = API.recordUSR(Field);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Field->getLocation());
    DocComment Comment;
    if (auto *RawComment =
            getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Field))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the struct field.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForField(Field);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Field);

    API.addRecordField(
        RecordRecord, Name, USR, Loc, AvailabilityInfo::createFromDecl(Field),
        Comment, Declaration, SubHeading, FieldKind, isInSystemHeader(Field));
  }
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitFieldDecl(const FieldDecl *Decl) {
  if (Decl->getDeclContext()->getDeclKind() == Decl::Record)
    return true;
  if (isa<ObjCIvarDecl>(Decl))
    return true;
  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the struct field.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForField(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  AccessControl Access = DeclarationFragmentsBuilder::getAccessControl(Decl);

  SmallString<128> ParentUSR;
  index::generateUSRForDecl(dyn_cast<CXXRecordDecl>(Decl->getDeclContext()),
                            ParentUSR);
  API.addCXXField(API.findRecordForUSR(ParentUSR), Name, USR, Loc,
                  AvailabilityInfo::createFromDecl(Decl), Comment, Declaration,
                  SubHeading, Access, isInSystemHeader(Decl));
  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitCXXConversionDecl(
    const CXXConversionDecl *Decl) {
  StringRef Name = API.copyString(Decl->getNameAsString());
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment =
          getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments, sub-heading, and signature for the method.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForConversionFunction(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  FunctionSignature Signature =
      DeclarationFragmentsBuilder::getFunctionSignature(Decl);
  AccessControl Access = DeclarationFragmentsBuilder::getAccessControl(Decl);

  SmallString<128> ParentUSR;
  index::generateUSRForDecl(dyn_cast<CXXRecordDecl>(Decl->getDeclContext()),
                            ParentUSR);
  if (Decl->isStatic())
    API.addCXXStaticMethod(API.findRecordForUSR(ParentUSR), Name, USR, Loc,
                           AvailabilityInfo::createFromDecl(Decl), Comment,
                           Declaration, SubHeading, Signature, Access,
                           isInSystemHeader(Decl));
  else
    API.addCXXInstanceMethod(API.findRecordForUSR(ParentUSR), Name, USR, Loc,
                             AvailabilityInfo::createFromDecl(Decl), Comment,
                             Declaration, SubHeading, Signature, Access,
                             isInSystemHeader(Decl));
  return true;
}

/// Collect API information for the Objective-C methods and associate with the
/// parent container.
template <typename Derived>
void ExtractAPIVisitorBase<Derived>::recordObjCMethods(
    ObjCContainerRecord *Container,
    const ObjCContainerDecl::method_range Methods) {
  for (const auto *Method : Methods) {
    // Don't record selectors for properties.
    if (Method->isPropertyAccessor())
      continue;

    StringRef Name = API.copyString(Method->getSelector().getAsString());
    StringRef USR = API.recordUSR(Method);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Method->getLocation());
    DocComment Comment;
    if (auto *RawComment =
            getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Method))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments, sub-heading, and signature for the method.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForObjCMethod(Method);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Method);
    FunctionSignature Signature =
        DeclarationFragmentsBuilder::getFunctionSignature(Method);

    API.addObjCMethod(Container, Name, USR, Loc,
                      AvailabilityInfo::createFromDecl(Method), Comment,
                      Declaration, SubHeading, Signature,
                      Method->isInstanceMethod(), isInSystemHeader(Method));
  }
}

template <typename Derived>
void ExtractAPIVisitorBase<Derived>::recordObjCProperties(
    ObjCContainerRecord *Container,
    const ObjCContainerDecl::prop_range Properties) {
  for (const auto *Property : Properties) {
    StringRef Name = Property->getName();
    StringRef USR = API.recordUSR(Property);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Property->getLocation());
    DocComment Comment;
    if (auto *RawComment =
            getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Property))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the property.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForObjCProperty(Property);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Property);

    StringRef GetterName =
        API.copyString(Property->getGetterName().getAsString());
    StringRef SetterName =
        API.copyString(Property->getSetterName().getAsString());

    // Get the attributes for property.
    unsigned Attributes = ObjCPropertyRecord::NoAttr;
    if (Property->getPropertyAttributes() &
        ObjCPropertyAttribute::kind_readonly)
      Attributes |= ObjCPropertyRecord::ReadOnly;

    API.addObjCProperty(
        Container, Name, USR, Loc, AvailabilityInfo::createFromDecl(Property),
        Comment, Declaration, SubHeading,
        static_cast<ObjCPropertyRecord::AttributeKind>(Attributes), GetterName,
        SetterName, Property->isOptional(),
        !(Property->getPropertyAttributes() &
          ObjCPropertyAttribute::kind_class),
        isInSystemHeader(Property));
  }
}

template <typename Derived>
void ExtractAPIVisitorBase<Derived>::recordObjCInstanceVariables(
    ObjCContainerRecord *Container,
    const llvm::iterator_range<
        DeclContext::specific_decl_iterator<ObjCIvarDecl>>
        Ivars) {
  for (const auto *Ivar : Ivars) {
    StringRef Name = Ivar->getName();
    StringRef USR = API.recordUSR(Ivar);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Ivar->getLocation());
    DocComment Comment;
    if (auto *RawComment =
            getDerivedExtractAPIVisitor().fetchRawCommentForDecl(Ivar))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the instance variable.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForField(Ivar);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Ivar);

    ObjCInstanceVariableRecord::AccessControl Access =
        Ivar->getCanonicalAccessControl();

    API.addObjCInstanceVariable(
        Container, Name, USR, Loc, AvailabilityInfo::createFromDecl(Ivar),
        Comment, Declaration, SubHeading, Access, isInSystemHeader(Ivar));
  }
}

template <typename Derived>
void ExtractAPIVisitorBase<Derived>::recordObjCProtocols(
    ObjCContainerRecord *Container,
    ObjCInterfaceDecl::protocol_range Protocols) {
  for (const auto *Protocol : Protocols)
    Container->Protocols.emplace_back(Protocol->getName(),
                                      API.recordUSR(Protocol));
}

} // namespace impl

/// The RecursiveASTVisitor to traverse symbol declarations and collect API
/// information.
template <typename Derived = void>
class ExtractAPIVisitor
    : public impl::ExtractAPIVisitorBase<std::conditional_t<
          std::is_same_v<Derived, void>, ExtractAPIVisitor<>, Derived>> {
  using Base = impl::ExtractAPIVisitorBase<std::conditional_t<
      std::is_same_v<Derived, void>, ExtractAPIVisitor<>, Derived>>;

public:
  ExtractAPIVisitor(ASTContext &Context, APISet &API) : Base(Context, API) {}

  bool shouldDeclBeIncluded(const Decl *D) const { return true; }
  const RawComment *fetchRawCommentForDecl(const Decl *D) const {
    return this->Context.getRawCommentForDeclNoCache(D);
  }
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_EXTRACT_API_VISITOR_H
