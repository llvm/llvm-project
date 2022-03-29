//===- ExtractAPI/ExtractAPIConsumer.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ExtractAPIAction, and ASTVisitor/Consumer to
/// collect API information.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RawCommentList.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "clang/ExtractAPI/FrontendActions.h"
#include "clang/ExtractAPI/Serialization/SymbolGraphSerializer.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace extractapi;

namespace {

/// The RecursiveASTVisitor to traverse symbol declarations and collect API
/// information.
class ExtractAPIVisitor : public RecursiveASTVisitor<ExtractAPIVisitor> {
public:
  ExtractAPIVisitor(ASTContext &Context, Language Lang)
      : Context(Context), API(Context.getTargetInfo().getTriple(), Lang) {}

  const APISet &getAPI() const { return API; }

  bool VisitVarDecl(const VarDecl *Decl) {
    // Skip function parameters.
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

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
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
    API.addGlobalVar(Name, USR, Loc, Availability, Linkage, Comment,
                     Declaration, SubHeading);
    return true;
  }

  bool VisitFunctionDecl(const FunctionDecl *Decl) {
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

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
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
    API.addFunction(Name, USR, Loc, Availability, Linkage, Comment, Declaration,
                    SubHeading, Signature);
    return true;
  }

  bool VisitEnumDecl(const EnumDecl *Decl) {
    if (!Decl->isComplete())
      return true;

    // Skip forward declaration.
    if (!Decl->isThisDeclarationADefinition())
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the enum.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForEnum(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    EnumRecord *EnumRecord = API.addEnum(Name, USR, Loc, Availability, Comment,
                                         Declaration, SubHeading);

    // Now collect information about the enumerators in this enum.
    recordEnumConstants(EnumRecord, Decl->enumerators());

    return true;
  }

  bool VisitRecordDecl(const RecordDecl *Decl) {
    if (!Decl->isCompleteDefinition())
      return true;

    // Skip C++ structs/classes/unions
    // TODO: support C++ records
    if (isa<CXXRecordDecl>(Decl))
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
    DocComment Comment;
    if (auto *RawComment = Context.getRawCommentForDeclNoCache(Decl))
      Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                              Context.getDiagnostics());

    // Build declaration fragments and sub-heading for the struct.
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForStruct(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

    StructRecord *StructRecord = API.addStruct(
        Name, USR, Loc, Availability, Comment, Declaration, SubHeading);

    // Now collect information about the fields in this struct.
    recordStructFields(StructRecord, Decl->fields());

    return true;
  }

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *Decl) {
    // Skip forward declaration for classes (@class)
    if (!Decl->isThisDeclarationADefinition())
      return true;

    // Collect symbol information.
    StringRef Name = Decl->getName();
    StringRef USR = API.recordUSR(Decl);
    PresumedLoc Loc =
        Context.getSourceManager().getPresumedLoc(Decl->getLocation());
    AvailabilityInfo Availability = getAvailability(Decl);
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

    ObjCInterfaceRecord *ObjCInterfaceRecord =
        API.addObjCInterface(Name, USR, Loc, Availability, Linkage, Comment,
                             Declaration, SubHeading, SuperClass);

    // Record all methods (selectors). This doesn't include automatically
    // synthesized property methods.
    recordObjCMethods(ObjCInterfaceRecord, Decl->methods());
    recordObjCProperties(ObjCInterfaceRecord, Decl->properties());
    recordObjCInstanceVariables(ObjCInterfaceRecord, Decl->ivars());
    recordObjCProtocols(ObjCInterfaceRecord, Decl->protocols());

    return true;
  }

private:
  /// Get availability information of the declaration \p D.
  AvailabilityInfo getAvailability(const Decl *D) const {
    StringRef PlatformName = Context.getTargetInfo().getPlatformName();

    AvailabilityInfo Availability;
    // Collect availability attributes from all redeclarations.
    for (const auto *RD : D->redecls()) {
      for (const auto *A : RD->specific_attrs<AvailabilityAttr>()) {
        if (A->getPlatform()->getName() != PlatformName)
          continue;
        Availability = AvailabilityInfo(A->getIntroduced(), A->getDeprecated(),
                                        A->getObsoleted(), A->getUnavailable(),
                                        /* UnconditionallyDeprecated */ false,
                                        /* UnconditionallyUnavailable */ false);
        break;
      }

      if (const auto *A = RD->getAttr<UnavailableAttr>())
        if (!A->isImplicit()) {
          Availability.Unavailable = true;
          Availability.UnconditionallyUnavailable = true;
        }

      if (const auto *A = RD->getAttr<DeprecatedAttr>())
        if (!A->isImplicit())
          Availability.UnconditionallyDeprecated = true;
    }

    return Availability;
  }

  /// Collect API information for the enum constants and associate with the
  /// parent enum.
  void recordEnumConstants(EnumRecord *EnumRecord,
                           const EnumDecl::enumerator_range Constants) {
    for (const auto *Constant : Constants) {
      // Collect symbol information.
      StringRef Name = Constant->getName();
      StringRef USR = API.recordUSR(Constant);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Constant->getLocation());
      AvailabilityInfo Availability = getAvailability(Constant);
      DocComment Comment;
      if (auto *RawComment = Context.getRawCommentForDeclNoCache(Constant))
        Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                                Context.getDiagnostics());

      // Build declaration fragments and sub-heading for the enum constant.
      DeclarationFragments Declaration =
          DeclarationFragmentsBuilder::getFragmentsForEnumConstant(Constant);
      DeclarationFragments SubHeading =
          DeclarationFragmentsBuilder::getSubHeading(Constant);

      API.addEnumConstant(EnumRecord, Name, USR, Loc, Availability, Comment,
                          Declaration, SubHeading);
    }
  }

  /// Collect API information for the struct fields and associate with the
  /// parent struct.
  void recordStructFields(StructRecord *StructRecord,
                          const RecordDecl::field_range Fields) {
    for (const auto *Field : Fields) {
      // Collect symbol information.
      StringRef Name = Field->getName();
      StringRef USR = API.recordUSR(Field);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Field->getLocation());
      AvailabilityInfo Availability = getAvailability(Field);
      DocComment Comment;
      if (auto *RawComment = Context.getRawCommentForDeclNoCache(Field))
        Comment = RawComment->getFormattedLines(Context.getSourceManager(),
                                                Context.getDiagnostics());

      // Build declaration fragments and sub-heading for the struct field.
      DeclarationFragments Declaration =
          DeclarationFragmentsBuilder::getFragmentsForField(Field);
      DeclarationFragments SubHeading =
          DeclarationFragmentsBuilder::getSubHeading(Field);

      API.addStructField(StructRecord, Name, USR, Loc, Availability, Comment,
                         Declaration, SubHeading);
    }
  }

  /// Collect API information for the Objective-C methods and associate with the
  /// parent container.
  void recordObjCMethods(ObjCContainerRecord *Container,
                         const ObjCContainerDecl::method_range Methods) {
    for (const auto *Method : Methods) {
      // Don't record selectors for properties.
      if (Method->isPropertyAccessor())
        continue;

      StringRef Name = API.copyString(Method->getSelector().getAsString());
      StringRef USR = API.recordUSR(Method);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Method->getLocation());
      AvailabilityInfo Availability = getAvailability(Method);
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

      API.addObjCMethod(Container, Name, USR, Loc, Availability, Comment,
                        Declaration, SubHeading, Signature,
                        Method->isInstanceMethod());
    }
  }

  void recordObjCProperties(ObjCContainerRecord *Container,
                            const ObjCContainerDecl::prop_range Properties) {
    for (const auto *Property : Properties) {
      StringRef Name = Property->getName();
      StringRef USR = API.recordUSR(Property);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Property->getLocation());
      AvailabilityInfo Availability = getAvailability(Property);
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
      if (Property->getPropertyAttributes() & ObjCPropertyAttribute::kind_class)
        Attributes |= ObjCPropertyRecord::Class;

      API.addObjCProperty(
          Container, Name, USR, Loc, Availability, Comment, Declaration,
          SubHeading,
          static_cast<ObjCPropertyRecord::AttributeKind>(Attributes),
          GetterName, SetterName, Property->isOptional());
    }
  }

  void recordObjCInstanceVariables(
      ObjCContainerRecord *Container,
      const llvm::iterator_range<
          DeclContext::specific_decl_iterator<ObjCIvarDecl>>
          Ivars) {
    for (const auto *Ivar : Ivars) {
      StringRef Name = Ivar->getName();
      StringRef USR = API.recordUSR(Ivar);
      PresumedLoc Loc =
          Context.getSourceManager().getPresumedLoc(Ivar->getLocation());
      AvailabilityInfo Availability = getAvailability(Ivar);
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

      API.addObjCInstanceVariable(Container, Name, USR, Loc, Availability,
                                  Comment, Declaration, SubHeading, Access);
    }
  }

  void recordObjCProtocols(ObjCContainerRecord *Container,
                           ObjCInterfaceDecl::protocol_range Protocols) {
    for (const auto *Protocol : Protocols)
      Container->Protocols.emplace_back(Protocol->getName(),
                                        API.recordUSR(Protocol));
  }

  ASTContext &Context;
  APISet API;
};

class ExtractAPIConsumer : public ASTConsumer {
public:
  ExtractAPIConsumer(ASTContext &Context, StringRef ProductName, Language Lang,
                     std::unique_ptr<raw_pwrite_stream> OS)
      : Visitor(Context, Lang), ProductName(ProductName), OS(std::move(OS)) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    // Use ExtractAPIVisitor to traverse symbol declarations in the context.
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());

    // Setup a SymbolGraphSerializer to write out collected API information in
    // the Symbol Graph format.
    // FIXME: Make the kind of APISerializer configurable.
    SymbolGraphSerializer SGSerializer(Visitor.getAPI(), ProductName);
    SGSerializer.serialize(*OS);
  }

private:
  ExtractAPIVisitor Visitor;
  std::string ProductName;
  std::unique_ptr<raw_pwrite_stream> OS;
};

} // namespace

std::unique_ptr<ASTConsumer>
ExtractAPIAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<raw_pwrite_stream> OS = CreateOutputFile(CI, InFile);
  if (!OS)
    return nullptr;
  return std::make_unique<ExtractAPIConsumer>(
      CI.getASTContext(), CI.getInvocation().getFrontendOpts().ProductName,
      CI.getFrontendOpts().Inputs.back().getKind().getLanguage(),
      std::move(OS));
}

bool ExtractAPIAction::PrepareToExecuteAction(CompilerInstance &CI) {
  auto &Inputs = CI.getFrontendOpts().Inputs;
  if (Inputs.empty())
    return true;

  auto Kind = Inputs[0].getKind();

  // Convert the header file inputs into a single input buffer.
  SmallString<256> HeaderContents;
  for (const FrontendInputFile &FIF : Inputs) {
    if (Kind.isObjectiveC())
      HeaderContents += "#import";
    else
      HeaderContents += "#include";
    HeaderContents += " \"";
    HeaderContents += FIF.getFile();
    HeaderContents += "\"\n";
  }

  Buffer = llvm::MemoryBuffer::getMemBufferCopy(HeaderContents,
                                                getInputBufferName());

  // Set that buffer up as our "real" input in the CompilerInstance.
  Inputs.clear();
  Inputs.emplace_back(Buffer->getMemBufferRef(), Kind, /*IsSystem*/ false);

  return true;
}

std::unique_ptr<raw_pwrite_stream>
ExtractAPIAction::CreateOutputFile(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<raw_pwrite_stream> OS =
      CI.createDefaultOutputFile(/*Binary=*/false, InFile, /*Extension=*/"json",
                                 /*RemoveFileOnSignal=*/false);
  if (!OS)
    return nullptr;
  return OS;
}
