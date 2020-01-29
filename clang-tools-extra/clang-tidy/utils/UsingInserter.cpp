//===---------- UsingInserter.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UsingInserter.h"

#include "ASTUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace tidy {
namespace utils {

using namespace ast_matchers;

static StringRef getUnqualifiedName(StringRef QualifiedName) {
  size_t LastSeparatorPos = QualifiedName.rfind("::");
  if (LastSeparatorPos == StringRef::npos)
    return QualifiedName;
  return QualifiedName.drop_front(LastSeparatorPos + 2);
}

UsingInserter::UsingInserter(const SourceManager &SourceMgr)
    : SourceMgr(SourceMgr) {}

Optional<FixItHint> UsingInserter::createUsingDeclaration(
    ASTContext &Context, const Stmt &Statement, StringRef QualifiedName) {
  StringRef UnqualifiedName = getUnqualifiedName(QualifiedName);
  const FunctionDecl *Function = getSurroundingFunction(Context, Statement);
  if (!Function)
    return None;

  if (AddedUsing.count(std::make_pair(Function, QualifiedName.str())) != 0)
    return None;

  SourceLocation InsertLoc = Lexer::getLocForEndOfToken(
      Function->getBody()->getBeginLoc(), 0, SourceMgr, Context.getLangOpts());

  // Only use using declarations in the main file, not in includes.
  if (SourceMgr.getFileID(InsertLoc) != SourceMgr.getMainFileID())
    return None;

  // FIXME: This declaration could be masked. Investigate if
  // there is a way to avoid using Sema.
  bool AlreadyHasUsingDecl =
      !match(stmt(hasAncestor(decl(has(usingDecl(hasAnyUsingShadowDecl(
                 hasTargetDecl(hasName(QualifiedName.str())))))))),
             Statement, Context)
           .empty();
  if (AlreadyHasUsingDecl) {
    AddedUsing.emplace(NameInFunction(Function, QualifiedName.str()));
    return None;
  }
  // Find conflicting declarations and references.
  auto ConflictingDecl = namedDecl(hasName(UnqualifiedName));
  bool HasConflictingDeclaration =
      !match(findAll(ConflictingDecl), *Function, Context).empty();
  bool HasConflictingDeclRef =
      !match(findAll(declRefExpr(to(ConflictingDecl))), *Function, Context)
           .empty();
  if (HasConflictingDeclaration || HasConflictingDeclRef)
    return None;

  std::string Declaration =
      (llvm::Twine("\nusing ") + QualifiedName + ";").str();

  AddedUsing.emplace(std::make_pair(Function, QualifiedName.str()));
  return FixItHint::CreateInsertion(InsertLoc, Declaration);
}

StringRef UsingInserter::getShortName(ASTContext &Context,
                                      const Stmt &Statement,
                                      StringRef QualifiedName) {
  const FunctionDecl *Function = getSurroundingFunction(Context, Statement);
  if (AddedUsing.count(NameInFunction(Function, QualifiedName.str())) != 0)
    return getUnqualifiedName(QualifiedName);
  return QualifiedName;
}

} // namespace utils
} // namespace tidy
} // namespace clang
