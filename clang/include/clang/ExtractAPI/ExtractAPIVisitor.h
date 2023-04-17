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

#include "llvm/ADT/FunctionExtras.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/TypedefUnderlyingTypeResolver.h"
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

  bool VisitRecordDecl(const RecordDecl *Decl);

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

  /// Collect API information for the struct fields and associate with the
  /// parent struct.
  void recordStructFields(StructRecord *StructRecord,
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
};

template <typename T>
static void modifyRecords(const T &Records, const StringRef &Name) {
  for (const auto &Record : Records) {
    if (Name == Record.second.get()->Name) {
      Record.second.get()->Declaration.removeLast();
      Record.second.get()
          ->Declaration
          .appendFront(" ", DeclarationFragments::FragmentKind::Text)
          .appendFront("typedef", DeclarationFragments::FragmentKind::Keyword,
                       "", nullptr)
          .append(" { ... } ", DeclarationFragments::FragmentKind::Text)
          .append(Name, DeclarationFragments::FragmentKind::Identifier)
          .append(";", DeclarationFragments::FragmentKind::Text);
      break;
    }
  }
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitVarDecl(const VarDecl *Decl) {
  // skip function parameters.
  if (isa<ParmVarDecl>(Decl))
    return true;

  // Skip non-global variables in records (struct/union/class).
  if (Decl->getDeclContext()->isRecord())
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

  // Add the global variable record to the API set.
  API.addGlobalVar(Name, USR, Loc, AvailabilitySet(Decl), Linkage, Comment,
                   Declaration, SubHeading, isInSystemHeader(Decl));
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
    for (auto P : Context.getParents(*Method)) {
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
    break;
  case FunctionDecl::TK_MemberSpecialization:
  case FunctionDecl::TK_FunctionTemplateSpecialization:
    if (auto *TemplateInfo = Decl->getTemplateSpecializationInfo()) {
      if (!TemplateInfo->isExplicitInstantiationOrSpecialization())
        return true;
    }
    break;
  case FunctionDecl::TK_FunctionTemplate:
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization:
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
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForFunction(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);
  FunctionSignature Signature =
      DeclarationFragmentsBuilder::getFunctionSignature(Decl);

  // Add the function record to the API set.
  API.addGlobalFunction(Name, USR, Loc, AvailabilitySet(Decl), Linkage, Comment,
                        Declaration, SubHeading, Signature,
                        isInSystemHeader(Decl));
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

  EnumRecord *EnumRecord =
      API.addEnum(API.copyString(Name), USR, Loc, AvailabilitySet(Decl),
                  Comment, Declaration, SubHeading, isInSystemHeader(Decl));

  // Now collect information about the enumerators in this enum.
  getDerivedExtractAPIVisitor().recordEnumConstants(EnumRecord,
                                                    Decl->enumerators());

  return true;
}

template <typename Derived>
bool ExtractAPIVisitorBase<Derived>::VisitRecordDecl(const RecordDecl *Decl) {
  // Skip C++ structs/classes/unions
  // TODO: support C++ records
  if (isa<CXXRecordDecl>(Decl))
    return true;

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
      DeclarationFragmentsBuilder::getFragmentsForStruct(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  StructRecord *StructRecord =
      API.addStruct(Name, USR, Loc, AvailabilitySet(Decl), Comment, Declaration,
                    SubHeading, isInSystemHeader(Decl));

  // Now collect information about the fields in this struct.
  getDerivedExtractAPIVisitor().recordStructFields(StructRecord,
                                                   Decl->fields());

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
      Name, USR, Loc, AvailabilitySet(Decl), Linkage, Comment, Declaration,
      SubHeading, SuperClass, isInSystemHeader(Decl));

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

  ObjCProtocolRecord *ObjCProtocolRecord =
      API.addObjCProtocol(Name, USR, Loc, AvailabilitySet(Decl), Comment,
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
        if (TagTy->getDecl()->isStruct()) {
          modifyRecords(API.getStructs(), Decl->getName());
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

  API.addTypedef(Name, USR, Loc, AvailabilitySet(Decl), Comment,
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

  ObjCCategoryRecord *ObjCCategoryRecord = API.addObjCCategory(
      Name, USR, Loc, AvailabilitySet(Decl), Comment, Declaration, SubHeading,
      Interface, isInSystemHeader(Decl));

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

    API.addEnumConstant(EnumRecord, Name, USR, Loc, AvailabilitySet(Constant),
                        Comment, Declaration, SubHeading,
                        isInSystemHeader(Constant));
  }
}

/// Collect API information for the struct fields and associate with the
/// parent struct.
template <typename Derived>
void ExtractAPIVisitorBase<Derived>::recordStructFields(
    StructRecord *StructRecord, const RecordDecl::field_range Fields) {
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

    API.addStructField(StructRecord, Name, USR, Loc, AvailabilitySet(Field),
                       Comment, Declaration, SubHeading,
                       isInSystemHeader(Field));
  }
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

    API.addObjCMethod(Container, Name, USR, Loc, AvailabilitySet(Method),
                      Comment, Declaration, SubHeading, Signature,
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
        Container, Name, USR, Loc, AvailabilitySet(Property), Comment,
        Declaration, SubHeading,
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

    API.addObjCInstanceVariable(Container, Name, USR, Loc,
                                AvailabilitySet(Ivar), Comment, Declaration,
                                SubHeading, Access, isInSystemHeader(Ivar));
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
