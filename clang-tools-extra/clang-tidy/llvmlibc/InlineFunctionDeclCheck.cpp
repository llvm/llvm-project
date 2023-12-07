//===-- InlineFunctionDeclCheck.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InlineFunctionDeclCheck.h"
#include "../utils/FileExtensionsUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_libc {

namespace {

const TemplateParameterList *
getLastTemplateParameterList(const FunctionDecl *FuncDecl) {
  const TemplateParameterList *ReturnList =
      FuncDecl->getDescribedTemplateParams();

  if (!ReturnList) {
    const unsigned NumberOfTemplateParameterLists =
        FuncDecl->getNumTemplateParameterLists();

    if (NumberOfTemplateParameterLists > 0)
      ReturnList = FuncDecl->getTemplateParameterList(
          NumberOfTemplateParameterLists - 1);
  }

  return ReturnList;
}

} // namespace

InlineFunctionDeclCheck::InlineFunctionDeclCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      HeaderFileExtensions(Context->getHeaderFileExtensions()) {}

void InlineFunctionDeclCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(decl(functionDecl()).bind("func_decl"), this);
}

void InlineFunctionDeclCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>("func_decl");

  // Consider only explicitly or implicitly inline functions.
  if (FuncDecl == nullptr || !FuncDecl->isInlined())
    return;

  SourceLocation SrcBegin = FuncDecl->getBeginLoc();

  // If we have a template parameter list, we need to skip that because the
  // LIBC_INLINE macro must be placed after that.
  if (const TemplateParameterList *TemplateParams =
          getLastTemplateParameterList(FuncDecl)) {
    SrcBegin = TemplateParams->getRAngleLoc();
    std::optional<Token> NextToken =
        utils::lexer::findNextTokenSkippingComments(
            SrcBegin, *Result.SourceManager, Result.Context->getLangOpts());
    if (NextToken)
      SrcBegin = NextToken->getLocation();
  }

  // Consider functions only in header files.
  if (!utils::isSpellingLocInHeaderFile(SrcBegin, *Result.SourceManager,
                                        HeaderFileExtensions))
    return;

  // Ignore lambda functions as they are internal and implicit.
  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(FuncDecl))
    if (MethodDecl->getParent()->isLambda())
      return;

  // Check if decl starts with LIBC_INLINE
  auto Loc = FullSourceLoc(Result.SourceManager->getFileLoc(SrcBegin),
                           *Result.SourceManager);
  llvm::StringRef SrcText = Loc.getBufferData().drop_front(Loc.getFileOffset());
  if (SrcText.starts_with("LIBC_INLINE"))
    return;

  diag(SrcBegin, "%0 must be tagged with the LIBC_INLINE macro; the macro "
                 "should be placed at the beginning of the declaration")
      << FuncDecl << FixItHint::CreateInsertion(Loc, "LIBC_INLINE ");
}

} // namespace clang::tidy::llvm_libc
