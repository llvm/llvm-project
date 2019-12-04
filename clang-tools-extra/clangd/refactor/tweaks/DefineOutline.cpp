//===--- DefineOutline.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "FindTarget.h"
#include "HeaderSourceSwitch.h"
#include "Logger.h"
#include "Path.h"
#include "Selection.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Types.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <string>

namespace clang {
namespace clangd {
namespace {

// Deduces the FunctionDecl from a selection. Requires either the function body
// or the function decl to be selected. Returns null if none of the above
// criteria is met.
// FIXME: This is shared with define inline, move them to a common header once
// we have a place for such.
const FunctionDecl *getSelectedFunction(const SelectionTree::Node *SelNode) {
  if (!SelNode)
    return nullptr;
  const ast_type_traits::DynTypedNode &AstNode = SelNode->ASTNode;
  if (const FunctionDecl *FD = AstNode.get<FunctionDecl>())
    return FD;
  if (AstNode.get<CompoundStmt>() &&
      SelNode->Selected == SelectionTree::Complete) {
    if (const SelectionTree::Node *P = SelNode->Parent)
      return P->ASTNode.get<FunctionDecl>();
  }
  return nullptr;
}

llvm::Optional<Path> getSourceFile(llvm::StringRef FileName,
                                   const Tweak::Selection &Sel) {
  if (auto Source = getCorrespondingHeaderOrSource(
          FileName,
          &Sel.AST.getSourceManager().getFileManager().getVirtualFileSystem()))
    return *Source;
  return getCorrespondingHeaderOrSource(FileName, Sel.AST, Sel.Index);
}

// Synthesize a DeclContext for TargetNS from CurContext. TargetNS must be empty
// for global namespace, and endwith "::" otherwise.
// Returns None if TargetNS is not a prefix of CurContext.
llvm::Optional<const DeclContext *>
findContextForNS(llvm::StringRef TargetNS, const DeclContext *CurContext) {
  assert(TargetNS.empty() || TargetNS.endswith("::"));
  // Skip any non-namespace contexts, e.g. TagDecls, functions/methods.
  CurContext = CurContext->getEnclosingNamespaceContext();
  // If TargetNS is empty, it means global ns, which is translation unit.
  if (TargetNS.empty()) {
    while (!CurContext->isTranslationUnit())
      CurContext = CurContext->getParent();
    return CurContext;
  }
  // Otherwise we need to drop any trailing namespaces from CurContext until
  // we reach TargetNS.
  std::string TargetContextNS =
      CurContext->isNamespace()
          ? llvm::cast<NamespaceDecl>(CurContext)->getQualifiedNameAsString()
          : "";
  TargetContextNS.append("::");

  llvm::StringRef CurrentContextNS(TargetContextNS);
  // If TargetNS is not a prefix of CurrentContext, there's no way to reach
  // it.
  if (!CurrentContextNS.startswith(TargetNS))
    return llvm::None;

  while (CurrentContextNS != TargetNS) {
    CurContext = CurContext->getParent();
    // These colons always exists since TargetNS is a prefix of
    // CurrentContextNS, it ends with "::" and they are not equal.
    CurrentContextNS = CurrentContextNS.take_front(
        CurrentContextNS.drop_back(2).rfind("::") + 2);
  }
  return CurContext;
}

// Returns source code for FD after applying Replacements.
// FIXME: Make the function take a parameter to return only the function body,
// afterwards it can be shared with define-inline code action.
llvm::Expected<std::string>
getFunctionSourceAfterReplacements(const FunctionDecl *FD,
                                   const tooling::Replacements &Replacements) {
  const auto &SM = FD->getASTContext().getSourceManager();
  auto OrigFuncRange = toHalfOpenFileRange(
      SM, FD->getASTContext().getLangOpts(), FD->getSourceRange());
  if (!OrigFuncRange)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Couldn't get range for function.");
  // Include template parameter list.
  if (auto *FTD = FD->getDescribedFunctionTemplate())
    OrigFuncRange->setBegin(FTD->getBeginLoc());

  // Get new begin and end positions for the qualified function definition.
  unsigned FuncBegin = SM.getFileOffset(OrigFuncRange->getBegin());
  unsigned FuncEnd = Replacements.getShiftedCodePosition(
      SM.getFileOffset(OrigFuncRange->getEnd()));

  // Trim the result to function definition.
  auto QualifiedFunc = tooling::applyAllReplacements(
      SM.getBufferData(SM.getMainFileID()), Replacements);
  if (!QualifiedFunc)
    return QualifiedFunc.takeError();
  return QualifiedFunc->substr(FuncBegin, FuncEnd - FuncBegin + 1);
}

// Creates a modified version of function definition that can be inserted at a
// different location, qualifies return value and function name to achieve that.
// Contains function signature, body and template parameters if applicable.
// No need to qualify parameters, as they are looked up in the context
// containing the function/method.
// FIXME: Qualify function name depending on the target context.
llvm::Expected<std::string>
getFunctionSourceCode(const FunctionDecl *FD, llvm::StringRef TargetNamespace) {
  auto &SM = FD->getASTContext().getSourceManager();
  auto TargetContext = findContextForNS(TargetNamespace, FD->getDeclContext());
  if (!TargetContext)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "define outline: couldn't find a context for target");

  llvm::Error Errors = llvm::Error::success();
  tooling::Replacements QualifierInsertions;

  // Finds the first unqualified name in function return type and qualifies it
  // to be valid in TargetContext.
  findExplicitReferences(FD, [&](ReferenceLoc Ref) {
    // It is enough to qualify the first qualifier, so skip references with a
    // qualifier. Also we can't do much if there are no targets or name is
    // inside a macro body.
    if (Ref.Qualifier || Ref.Targets.empty() || Ref.NameLoc.isMacroID())
      return;
    // Qualify return type
    if (Ref.NameLoc != FD->getReturnTypeSourceRange().getBegin())
      return;

    for (const NamedDecl *ND : Ref.Targets) {
      if (ND->getDeclContext() != Ref.Targets.front()->getDeclContext()) {
        elog("Targets from multiple contexts: {0}, {1}",
             printQualifiedName(*Ref.Targets.front()), printQualifiedName(*ND));
        return;
      }
    }
    const NamedDecl *ND = Ref.Targets.front();
    const std::string Qualifier =
        getQualification(FD->getASTContext(), *TargetContext,
                         SM.getLocForStartOfFile(SM.getMainFileID()), ND);
    if (auto Err = QualifierInsertions.add(
            tooling::Replacement(SM, Ref.NameLoc, 0, Qualifier)))
      Errors = llvm::joinErrors(std::move(Errors), std::move(Err));
  });

  if (Errors)
    return std::move(Errors);
  return getFunctionSourceAfterReplacements(FD, QualifierInsertions);
}

struct InsertionPoint {
  std::string EnclosingNamespace;
  size_t Offset;
};
// Returns the most natural insertion point for \p QualifiedName in \p Contents.
// This currently cares about only the namespace proximity, but in feature it
// should also try to follow ordering of declarations. For example, if decls
// come in order `foo, bar, baz` then this function should return some point
// between foo and baz for inserting bar.
llvm::Expected<InsertionPoint>
getInsertionPoint(llvm::StringRef Contents, llvm::StringRef QualifiedName,
                  const format::FormatStyle &Style) {
  auto Region = getEligiblePoints(Contents, QualifiedName, Style);

  assert(!Region.EligiblePoints.empty());
  // FIXME: This selection can be made smarter by looking at the definition
  // locations for adjacent decls to Source. Unfortunately psudeo parsing in
  // getEligibleRegions only knows about namespace begin/end events so we
  // can't match function start/end positions yet.
  auto Offset = positionToOffset(Contents, Region.EligiblePoints.back());
  if (!Offset)
    return Offset.takeError();
  return InsertionPoint{Region.EnclosingNamespace, *Offset};
}

/// Moves definition of a function/method to an appropriate implementation file.
///
/// Before:
/// a.h
///   void foo() { return; }
/// a.cc
///   #include "a.h"
///
/// ----------------
///
/// After:
/// a.h
///   void foo();
/// a.cc
///   #include "a.h"
///   void foo() { return; }
class DefineOutline : public Tweak {
public:
  const char *id() const override;

  bool hidden() const override { return true; }
  Intent intent() const override { return Intent::Refactor; }
  std::string title() const override {
    return "Move function body to out-of-line.";
  }

  bool prepare(const Selection &Sel) override {
    // Bail out if we are not in a header file.
    // FIXME: We might want to consider moving method definitions below class
    // definition even if we are inside a source file.
    if (!isHeaderFile(Sel.AST.getSourceManager().getFilename(Sel.Cursor),
                      Sel.AST.getASTContext().getLangOpts()))
      return false;

    Source = getSelectedFunction(Sel.ASTSelection.commonAncestor());
    // Bail out if the selection is not a in-line function definition.
    if (!Source || !Source->doesThisDeclarationHaveABody() ||
        Source->isOutOfLine())
      return false;

    // Bail out in templated classes, as it is hard to spell the class name, i.e
    // if the template parameter is unnamed.
    if (auto *MD = llvm::dyn_cast<CXXMethodDecl>(Source)) {
      if (MD->getParent()->isTemplated())
        return false;
    }

    // Note that we don't check whether an implementation file exists or not in
    // the prepare, since performing disk IO on each prepare request might be
    // expensive.
    return true;
  }

  Expected<Effect> apply(const Selection &Sel) override {
    const SourceManager &SM = Sel.AST.getSourceManager();
    auto MainFileName =
        getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
    if (!MainFileName)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Couldn't get absolute path for mainfile.");

    auto CCFile = getSourceFile(*MainFileName, Sel);
    if (!CCFile)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Couldn't find a suitable implementation file.");

    auto &FS =
        Sel.AST.getSourceManager().getFileManager().getVirtualFileSystem();
    auto Buffer = FS.getBufferForFile(*CCFile);
    // FIXME: Maybe we should consider creating the implementation file if it
    // doesn't exist?
    if (!Buffer)
      return llvm::createStringError(Buffer.getError(),
                                     Buffer.getError().message());
    auto Contents = Buffer->get()->getBuffer();
    auto InsertionPoint =
        getInsertionPoint(Contents, Source->getQualifiedNameAsString(),
                          getFormatStyleForFile(*CCFile, Contents, &FS));
    if (!InsertionPoint)
      return InsertionPoint.takeError();

    auto FuncDef =
        getFunctionSourceCode(Source, InsertionPoint->EnclosingNamespace);
    if (!FuncDef)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Couldn't get full source for function definition.");

    SourceManagerForFile SMFF(*CCFile, Contents);
    const tooling::Replacement InsertFunctionDef(
        *CCFile, InsertionPoint->Offset, 0, *FuncDef);
    auto Effect = Effect::mainFileEdit(
        SMFF.get(), tooling::Replacements(InsertFunctionDef));
    if (!Effect)
      return Effect.takeError();

    // FIXME: We should also get rid of inline qualifier.
    const tooling::Replacement DeleteFuncBody(
        Sel.AST.getSourceManager(),
        CharSourceRange::getTokenRange(
            *toHalfOpenFileRange(SM, Sel.AST.getASTContext().getLangOpts(),
                                 Source->getBody()->getSourceRange())),
        ";");
    auto HeaderFE = Effect::fileEdit(SM, SM.getMainFileID(),
                                     tooling::Replacements(DeleteFuncBody));
    if (!HeaderFE)
      return HeaderFE.takeError();

    Effect->ApplyEdits.try_emplace(HeaderFE->first,
                                   std::move(HeaderFE->second));
    return std::move(*Effect);
  }

private:
  const FunctionDecl *Source = nullptr;
};

REGISTER_TWEAK(DefineOutline);

} // namespace
} // namespace clangd
} // namespace clang
