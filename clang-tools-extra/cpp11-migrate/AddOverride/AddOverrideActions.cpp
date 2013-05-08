//===-- AddOverride/AddOverrideActions.cpp - add C++11 override-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the definition of the AddOverrideFixer class
///  which is used as an ASTMatcher callback.
///
//===----------------------------------------------------------------------===//

#include "AddOverrideActions.h"
#include "AddOverrideMatchers.h"

#include "clang/Basic/CharInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

void AddOverrideFixer::run(const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;

  const CXXMethodDecl *M = Result.Nodes.getDeclAs<CXXMethodDecl>(MethodId);
  assert(M && "Bad Callback. No node provided");

  // First check that there isn't already an override attribute.
  if (!M->hasAttr<OverrideAttr>()) {
    if (M->getLocStart().isFileID()) {
      SourceLocation StartLoc;
      if (M->hasInlineBody()) {
        // Start at the beginning of the body and rewind back to the last
        // non-whitespace character. We will insert the override keyword
        // after that character.
        // FIXME: This transform won't work if there is a comment between
        // the end of the function prototype and the start of the body.
        StartLoc = M->getBody()->getLocStart();
        do {
          StartLoc = StartLoc.getLocWithOffset(-1);
        } while (isWhitespace(*FullSourceLoc(StartLoc, SM).getCharacterData()));
        StartLoc = StartLoc.getLocWithOffset(1);
      } else {
        StartLoc = SM.getSpellingLoc(M->getLocEnd());
        StartLoc = Lexer::getLocForEndOfToken(StartLoc, 0, SM, LangOptions());
      }
      Replace.insert(tooling::Replacement(SM, StartLoc, 0, " override"));
      ++AcceptedChanges;
    }
  }
}

