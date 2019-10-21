//===--- DefineOutline.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderSourceSwitch.h"
#include "Path.h"
#include "Selection.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Types.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstddef>

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

// Creates a modified version of function definition that can be inserted at a
// different location. Contains both function signature and body.
llvm::Optional<llvm::StringRef> getFunctionSourceCode(const FunctionDecl *FD) {
  auto &SM = FD->getASTContext().getSourceManager();
  auto CharRange = toHalfOpenFileRange(SM, FD->getASTContext().getLangOpts(),
                                       FD->getSourceRange());
  if (!CharRange)
    return llvm::None;
  // Include template parameter list.
  if (auto *FTD = FD->getDescribedFunctionTemplate())
    CharRange->setBegin(FTD->getBeginLoc());

  // FIXME: Qualify return type.
  // FIXME: Qualify function name depending on the target context.
  return toSourceCode(SM, *CharRange);
}

// Returns the most natural insertion point for \p QualifiedName in \p Contents.
// This currently cares about only the namespace proximity, but in feature it
// should also try to follow ordering of declarations. For example, if decls
// come in order `foo, bar, baz` then this function should return some point
// between foo and baz for inserting bar.
llvm::Expected<size_t> getInsertionOffset(llvm::StringRef Contents,
                                          llvm::StringRef QualifiedName,
                                          const format::FormatStyle &Style) {
  auto Region = getEligiblePoints(Contents, QualifiedName, Style);

  assert(!Region.EligiblePoints.empty());
  // FIXME: This selection can be made smarter by looking at the definition
  // locations for adjacent decls to Source. Unfortunately psudeo parsing in
  // getEligibleRegions only knows about namespace begin/end events so we
  // can't match function start/end positions yet.
  auto InsertionPoint = Region.EligiblePoints.back();
  return positionToOffset(Contents, InsertionPoint);
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
    auto InsertionOffset =
        getInsertionOffset(Contents, Source->getQualifiedNameAsString(),
                           getFormatStyleForFile(*CCFile, Contents, &FS));
    if (!InsertionOffset)
      return InsertionOffset.takeError();

    auto FuncDef = getFunctionSourceCode(Source);
    if (!FuncDef)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Couldn't get full source for function definition.");

    SourceManagerForFile SMFF(*CCFile, Contents);
    const tooling::Replacement InsertFunctionDef(*CCFile, *InsertionOffset, 0,
                                                 *FuncDef);
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
