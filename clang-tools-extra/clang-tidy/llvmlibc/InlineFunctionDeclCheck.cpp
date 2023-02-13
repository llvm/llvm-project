//===-- InlineFunctionDeclCheck.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InlineFunctionDeclCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_libc {

InlineFunctionDeclCheck::InlineFunctionDeclCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawStringHeaderFileExtensions(Options.getLocalOrGlobal(
          "HeaderFileExtensions", utils::defaultHeaderFileExtensions())) {
  if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                  HeaderFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
    this->configurationDiag("Invalid header file extension: '%0'")
        << RawStringHeaderFileExtensions;
  }
}

void InlineFunctionDeclCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(decl(functionDecl()).bind("func_decl"), this);
}

void InlineFunctionDeclCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
  if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                  HeaderFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
    this->configurationDiag("Invalid header file extension: '%0'")
        << RawStringHeaderFileExtensions;
  }
}

void InlineFunctionDeclCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>("func_decl");

  // Consider only explicitly or implicitly inline functions.
  if (FuncDecl == nullptr || !FuncDecl->isInlined())
    return;

  SourceLocation SrcBegin = FuncDecl->getBeginLoc();
  // Consider functions only in header files.
  if (!utils::isSpellingLocInHeaderFile(SrcBegin, *Result.SourceManager,
                                        HeaderFileExtensions))
    return;

  // Check if decl starts with LIBC_INLINE
  auto Loc = FullSourceLoc(Result.SourceManager->getFileLoc(SrcBegin),
                           *Result.SourceManager);
  llvm::StringRef SrcText = Loc.getBufferData().drop_front(Loc.getFileOffset());
  if (SrcText.starts_with("LIBC_INLINE"))
    return;

  diag(SrcBegin, "%0 must be tagged with the LIBC_INLINE macro; the macro "
                 "should be placed at the beginning of the declaration")
      << FuncDecl;
}

} // namespace clang::tidy::llvm_libc
