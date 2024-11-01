//===- RedundantStringInitCheck.cpp - clang-tidy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantStringInitCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang {
namespace tidy {
namespace readability {

const char DefaultStringNames[] =
    "::std::basic_string_view;::std::basic_string";

static std::vector<StringRef> removeNamespaces(ArrayRef<StringRef> Names) {
  std::vector<StringRef> Result;
  Result.reserve(Names.size());
  for (StringRef Name : Names) {
    StringRef::size_type ColonPos = Name.rfind(':');
    Result.push_back(
        Name.drop_front(ColonPos == StringRef::npos ? 0 : ColonPos + 1));
  }
  return Result;
}

static const CXXConstructExpr *
getConstructExpr(const CXXCtorInitializer &CtorInit) {
  const Expr *InitExpr = CtorInit.getInit();
  if (const auto *CleanUpExpr = dyn_cast<ExprWithCleanups>(InitExpr))
    InitExpr = CleanUpExpr->getSubExpr();
  return dyn_cast<CXXConstructExpr>(InitExpr);
}

static llvm::Optional<SourceRange>
getConstructExprArgRange(const CXXConstructExpr &Construct) {
  SourceLocation B, E;
  for (const Expr *Arg : Construct.arguments()) {
    if (B.isInvalid())
      B = Arg->getBeginLoc();
    if (Arg->getEndLoc().isValid())
      E = Arg->getEndLoc();
  }
  if (B.isInvalid() || E.isInvalid())
    return std::nullopt;
  return SourceRange(B, E);
}

RedundantStringInitCheck::RedundantStringInitCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringNames(utils::options::parseStringList(
          Options.get("StringNames", DefaultStringNames))) {}

void RedundantStringInitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringNames", DefaultStringNames);
}

void RedundantStringInitCheck::registerMatchers(MatchFinder *Finder) {
  const auto HasStringTypeName = hasAnyName(StringNames);
  const auto HasStringCtorName = hasAnyName(removeNamespaces(StringNames));

  // Match string constructor.
  const auto StringConstructorExpr = expr(
      anyOf(cxxConstructExpr(argumentCountIs(1),
                             hasDeclaration(cxxMethodDecl(HasStringCtorName))),
            // If present, the second argument is the alloc object which must
            // not be present explicitly.
            cxxConstructExpr(argumentCountIs(2),
                             hasDeclaration(cxxMethodDecl(HasStringCtorName)),
                             hasArgument(1, cxxDefaultArgExpr()))));

  // Match a string constructor expression with an empty string literal.
  const auto EmptyStringCtorExpr = cxxConstructExpr(
      StringConstructorExpr,
      hasArgument(0, ignoringParenImpCasts(stringLiteral(hasSize(0)))));

  const auto EmptyStringCtorExprWithTemporaries =
      cxxConstructExpr(StringConstructorExpr,
                       hasArgument(0, ignoringImplicit(EmptyStringCtorExpr)));

  const auto StringType = hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(HasStringTypeName)))));
  const auto EmptyStringInit = traverse(
      TK_AsIs, expr(ignoringImplicit(anyOf(
                   EmptyStringCtorExpr, EmptyStringCtorExprWithTemporaries))));

  // Match a variable declaration with an empty string literal as initializer.
  // Examples:
  //     string foo = "";
  //     string bar("");
  Finder->addMatcher(
      traverse(TK_AsIs,
               namedDecl(varDecl(StringType, hasInitializer(EmptyStringInit))
                             .bind("vardecl"),
                         unless(parmVarDecl()))),
      this);
  // Match a field declaration with an empty string literal as initializer.
  Finder->addMatcher(
      namedDecl(fieldDecl(StringType, hasInClassInitializer(EmptyStringInit))
                    .bind("fieldDecl")),
      this);
  // Matches Constructor Initializers with an empty string literal as
  // initializer.
  // Examples:
  //     Foo() : SomeString("") {}
  Finder->addMatcher(
      cxxCtorInitializer(
          isWritten(),
          forField(allOf(StringType, optionally(hasInClassInitializer(
                                         EmptyStringInit.bind("empty_init"))))),
          withInitializer(EmptyStringInit))
          .bind("ctorInit"),
      this);
}

void RedundantStringInitCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *VDecl = Result.Nodes.getNodeAs<VarDecl>("vardecl")) {
    // VarDecl's getSourceRange() spans 'string foo = ""' or 'string bar("")'.
    // So start at getLocation() to span just 'foo = ""' or 'bar("")'.
    SourceRange ReplaceRange(VDecl->getLocation(), VDecl->getEndLoc());
    diag(VDecl->getLocation(), "redundant string initialization")
        << FixItHint::CreateReplacement(ReplaceRange, VDecl->getName());
  }
  if (const auto *FDecl = Result.Nodes.getNodeAs<FieldDecl>("fieldDecl")) {
    // FieldDecl's getSourceRange() spans 'string foo = ""'.
    // So start at getLocation() to span just 'foo = ""'.
    SourceRange ReplaceRange(FDecl->getLocation(), FDecl->getEndLoc());
    diag(FDecl->getLocation(), "redundant string initialization")
        << FixItHint::CreateReplacement(ReplaceRange, FDecl->getName());
  }
  if (const auto *CtorInit =
          Result.Nodes.getNodeAs<CXXCtorInitializer>("ctorInit")) {
    if (const FieldDecl *Member = CtorInit->getMember()) {
      if (!Member->hasInClassInitializer() ||
          Result.Nodes.getNodeAs<Expr>("empty_init")) {
        // The String isn't declared in the class with an initializer or its
        // declared with a redundant initializer, which will be removed. Either
        // way the string will be default initialized, therefore we can remove
        // the constructor initializer entirely.
        diag(CtorInit->getMemberLocation(), "redundant string initialization")
            << FixItHint::CreateRemoval(CtorInit->getSourceRange());
        return;
      }
    }
    const CXXConstructExpr *Construct = getConstructExpr(*CtorInit);
    if (!Construct)
      return;
    if (llvm::Optional<SourceRange> RemovalRange =
            getConstructExprArgRange(*Construct))
      diag(CtorInit->getMemberLocation(), "redundant string initialization")
          << FixItHint::CreateRemoval(*RemovalRange);
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
