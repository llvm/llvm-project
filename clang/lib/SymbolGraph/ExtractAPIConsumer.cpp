//===- ExtractAPIConsumer.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the ExtractAPI AST visitor to collect API information.
///
//===----------------------------------------------------------------------===//
//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RawCommentList.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/SymbolGraph/API.h"
#include "clang/SymbolGraph/AvailabilityInfo.h"
#include "clang/SymbolGraph/DeclarationFragments.h"
#include "clang/SymbolGraph/FrontendActions.h"
#include "clang/SymbolGraph/Serialization.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace symbolgraph;

namespace {
class ExtractAPIVisitor : public RecursiveASTVisitor<ExtractAPIVisitor> {
public:
  explicit ExtractAPIVisitor(ASTContext &Context)
      : Context(Context),
        API(Context.getTargetInfo().getTriple(), Context.getLangOpts()) {}

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
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForVar(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);

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
    DeclarationFragments Declaration =
        DeclarationFragmentsBuilder::getFragmentsForFunction(Decl);
    DeclarationFragments SubHeading =
        DeclarationFragmentsBuilder::getSubHeading(Decl);
    FunctionSignature Signature =
        DeclarationFragmentsBuilder::getFunctionSignature(Decl);

    API.addFunction(Name, USR, Loc, Availability, Linkage, Comment, Declaration,
                    SubHeading, Signature);
    return true;
  }

private:
  AvailabilityInfo getAvailability(const Decl *D) const {
    StringRef PlatformName = Context.getTargetInfo().getPlatformName();

    AvailabilityInfo Availability;
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

  ASTContext &Context;
  APISet API;
};

class ExtractAPIConsumer : public ASTConsumer {
public:
  ExtractAPIConsumer(ASTContext &Context, std::unique_ptr<raw_pwrite_stream> OS)
      : Visitor(Context), OS(std::move(OS)) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    Serializer Serializer(Visitor.getAPI());
    Serializer.serialize(*OS);
  }

private:
  ExtractAPIVisitor Visitor;
  std::unique_ptr<raw_pwrite_stream> OS;
};
} // namespace

std::unique_ptr<ASTConsumer>
ExtractAPIAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<raw_pwrite_stream> OS = CreateOutputFile(CI, InFile);
  if (!OS)
    return nullptr;
  return std::make_unique<ExtractAPIConsumer>(CI.getASTContext(),
                                              std::move(OS));
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
