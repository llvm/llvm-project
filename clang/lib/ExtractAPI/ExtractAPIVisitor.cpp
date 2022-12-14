//===- ExtractAPI/ExtractAPIVisitor.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ExtractAPIVisitor an ASTVisitor to collect API
/// information.
///
//===----------------------------------------------------------------------===//

#include "clang/ExtractAPI/ExtractAPIVisitor.h"

#include "TypedefUnderlyingTypeResolver.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendOptions.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace extractapi;

namespace {

StringRef getTypedefName(const TagDecl *Decl) {
  if (const auto *TypedefDecl = Decl->getTypedefNameForAnonDecl())
    return TypedefDecl->getName();

  return {};
}

template <class DeclTy>
bool isInSystemHeader(const ASTContext &Context, const DeclTy *D) {
  return Context.getSourceManager().isInSystemHeader(D->getLocation());
}

} // namespace

bool ExtractAPIVisitor::VisitVarDecl(const VarDecl *Decl) {
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

  if (!LocationChecker(Decl->getLocation()))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the variable.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForVar(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  // Add the global variable record to the API set.
  API.addGlobalVar(Name, USR, Loc, AvailabilitySet(Decl), Linkage, Comment,
                   Declaration, SubHeading, isInSystemHeader(Context, Decl));
  return true;
}

bool ExtractAPIVisitor::VisitFunctionDecl(const FunctionDecl *Decl) {
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Decl)) {
    // Skip member function in class templates.
    if (Method->getParent()->getDescribedClassTemplate() != nullptr)
      return true;

    // Skip methods in records.
    for (auto P : Context.getParents(*Method)) {
      if (P.get<CXXRecordDecl>())
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

  if (!LocationChecker(Decl->getLocation()))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
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
                        isInSystemHeader(Context, Decl));
  return true;
}

bool ExtractAPIVisitor::VisitEnumDecl(const EnumDecl *Decl) {
  if (!Decl->isComplete())
    return true;

  // Skip forward declaration.
  if (!Decl->isThisDeclarationADefinition())
    return true;

  if (!LocationChecker(Decl->getLocation()))
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
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the enum.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForEnum(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  EnumRecord *EnumRecord = API.addEnum(
      API.copyString(Name), USR, Loc, AvailabilitySet(Decl), Comment,
      Declaration, SubHeading, isInSystemHeader(Context, Decl));

  // Now collect information about the enumerators in this enum.
  recordEnumConstants(EnumRecord, Decl->enumerators());

  return true;
}

bool ExtractAPIVisitor::VisitRecordDecl(const RecordDecl *Decl) {
  if (!Decl->isCompleteDefinition())
    return true;

  // Skip C++ structs/classes/unions
  // TODO: support C++ records
  if (isa<CXXRecordDecl>(Decl))
    return true;

  if (!LocationChecker(Decl->getLocation()))
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
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the struct.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForStruct(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  StructRecord *StructRecord =
      API.addStruct(Name, USR, Loc, AvailabilitySet(Decl), Comment, Declaration,
                    SubHeading, isInSystemHeader(Context, Decl));

  // Now collect information about the fields in this struct.
  recordStructFields(StructRecord, Decl->fields());

  return true;
}

bool ExtractAPIVisitor::VisitObjCInterfaceDecl(const ObjCInterfaceDecl *Decl) {
  // Skip forward declaration for classes (@class)
  if (!Decl->isThisDeclarationADefinition())
    return true;

  if (!LocationChecker(Decl->getLocation()))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  LinkageInfo Linkage = Decl->getLinkageAndVisibility();
  DocComment Comment;
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
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
      SubHeading, SuperClass, isInSystemHeader(Context, Decl));

  // Record all methods (selectors). This doesn't include automatically
  // synthesized property methods.
  recordObjCMethods(ObjCInterfaceRecord, Decl->methods());
  recordObjCProperties(ObjCInterfaceRecord, Decl->properties());
  recordObjCInstanceVariables(ObjCInterfaceRecord, Decl->ivars());
  recordObjCProtocols(ObjCInterfaceRecord, Decl->protocols());

  return true;
}

bool ExtractAPIVisitor::VisitObjCProtocolDecl(const ObjCProtocolDecl *Decl) {
  // Skip forward declaration for protocols (@protocol).
  if (!Decl->isThisDeclarationADefinition())
    return true;

  if (!LocationChecker(Decl->getLocation()))
    return true;

  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  // Build declaration fragments and sub-heading for the protocol.
  DeclarationFragments Declaration =
      DeclarationFragmentsBuilder::getFragmentsForObjCProtocol(Decl);
  DeclarationFragments SubHeading =
      DeclarationFragmentsBuilder::getSubHeading(Decl);

  ObjCProtocolRecord *ObjCProtocolRecord = API.addObjCProtocol(
      Name, USR, Loc, AvailabilitySet(Decl), Comment, Declaration, SubHeading,
      isInSystemHeader(Context, Decl));

  recordObjCMethods(ObjCProtocolRecord, Decl->methods());
  recordObjCProperties(ObjCProtocolRecord, Decl->properties());
  recordObjCProtocols(ObjCProtocolRecord, Decl->protocols());

  return true;
}

bool ExtractAPIVisitor::VisitTypedefNameDecl(const TypedefNameDecl *Decl) {
  // Skip ObjC Type Parameter for now.
  if (isa<ObjCTypeParamDecl>(Decl))
    return true;

  if (!Decl->isDefinedOutsideFunctionOrMethod())
    return true;

  if (!LocationChecker(Decl->getLocation()))
    return true;

  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  DocComment Comment;
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
    Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                            Context.getDiagnostics());

  QualType Type = Decl->getUnderlyingType();
  SymbolReference SymRef =
      TypedefUnderlyingTypeResolver(Context).getSymbolReferenceForType(Type,
                                                                       API);

  API.addTypedef(Name, USR, Loc, AvailabilitySet(Decl), Comment,
                 DeclarationFragmentsBuilder::getFragmentsForTypedef(Decl),
                 DeclarationFragmentsBuilder::getSubHeading(Decl), SymRef,
                 isInSystemHeader(Context, Decl));

  return true;
}

bool ExtractAPIVisitor::VisitObjCCategoryDecl(const ObjCCategoryDecl *Decl) {
  // Collect symbol information.
  StringRef Name = Decl->getName();
  StringRef USR = API.recordUSR(Decl);
  PresumedLoc Loc =
      Context.getSourceManager().getPresumedLoc(Decl->getLocation());
  DocComment Comment;
  if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
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
      Interface, isInSystemHeader(Context, Decl));

  recordObjCMethods(ObjCCategoryRecord, Decl->methods());
  recordObjCProperties(ObjCCategoryRecord, Decl->properties());
  recordObjCInstanceVariables(ObjCCategoryRecord, Decl->ivars());
  recordObjCProtocols(ObjCCategoryRecord, Decl->protocols());

  return true;
}

/// Collect API information for the enum constants and associate with the
/// parent enum.
void ExtractAPIVisitor::recordEnumConstants(
    EnumRecord *EnumRecord, const EnumDecl::enumerator_range Constants) {
  for (const auto *Constant : Constants) {
    // Collect symbol information.
    StringRef Name = Constant->getName();
    StringRef USR = API.recordUSR(Constant);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Constant->getLocation());
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Constant))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the enum constant.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForEnumConstant(Constant);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Constant);

    API.addEnumConstant(EnumRecord, Name, USR, Loc, AvailabilitySet(Constant),
                        Comment, Declaration, SubHeading,
                        isInSystemHeader(Context, Constant));
  }
}

/// Collect API information for the struct fields and associate with the
/// parent struct.
void ExtractAPIVisitor::recordStructFields(
    StructRecord *StructRecord, const RecordDecl::field_range Fields) {
  for (const auto *Field : Fields) {
    // Collect symbol information.
    StringRef Name = Field->getName();
    StringRef USR = API.recordUSR(Field);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Field->getLocation());
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Field))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the struct field.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForField(Field);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Field);

    API.addStructField(StructRecord, Name, USR, Loc, AvailabilitySet(Field),
                       Comment, Declaration, SubHeading,
                       isInSystemHeader(Context, Field));
  }
}

/// Collect API information for the Objective-C methods and associate with the
/// parent container.
void ExtractAPIVisitor::recordObjCMethods(
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
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Method))
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
                      Method->isInstanceMethod(),
                      isInSystemHeader(Context, Method));
  }
}

void ExtractAPIVisitor::recordObjCProperties(
    ObjCContainerRecord *Container,
    const ObjCContainerDecl::prop_range Properties) {
  for (const auto *Property : Properties) {
    StringRef Name = Property->getName();
    StringRef USR = API.recordUSR(Property);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Property->getLocation());
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Property))
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
        isInSystemHeader(Context, Property));
  }
}

void ExtractAPIVisitor::recordObjCInstanceVariables(
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
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Ivar))
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
        Container, Name, USR, Loc, AvailabilitySet(Ivar), Comment, Declaration,
        SubHeading, Access, isInSystemHeader(Context, Ivar));
  }
}

void ExtractAPIVisitor::recordObjCProtocols(
    ObjCContainerRecord *Container,
    ObjCInterfaceDecl::protocol_range Protocols) {
  for (const auto *Protocol : Protocols)
    Container->Protocols.emplace_back(Protocol->getName(),
                                      API.recordUSR(Protocol));
}
