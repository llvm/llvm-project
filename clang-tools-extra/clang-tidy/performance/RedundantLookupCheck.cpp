//===--- RedundantLookupCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantLookupCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static constexpr auto DefaultContainerNameRegex = "set|map";

static const llvm::StringRef DefaultLookupMethodNames =
    llvm::StringLiteral( //
        "at;"
        "contains;"
        "count;"
        "find_as;"
        "find;"
        // These are tricky, as they take the "key" at different places.
        // They sometimes bundle up the key and the value together in a pair.
        //   "emplace;"
        //   "insert_or_assign;"
        //   "insert;"
        //   "try_emplace;"
        )
        .drop_back(); // Drops the last semicolon.

RedundantLookupCheck::RedundantLookupCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ContainerNameRegex(
          Options.get("ContainerNameRegex", DefaultContainerNameRegex)),
      LookupMethodNames(utils::options::parseStringList(
          Options.get("LookupMethodNames", DefaultLookupMethodNames))) {}

void RedundantLookupCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ContainerNameRegex", ContainerNameRegex);
  Options.store(Opts, "LookupMethodNames",
                utils::options::serializeStringList(LookupMethodNames));
}

namespace {
/// Checks if any of the ends of the source range is in a macro expansion.
AST_MATCHER(Expr, hasMacroSourceRange) {
  SourceRange R = Node.getSourceRange();
  return R.getBegin().isMacroID() || R.getEnd().isMacroID();
}
} // namespace

static constexpr const char *ObjKey = "obj";
static constexpr const char *LookupKey = "key";
static constexpr const char *LookupCallKey = "lookup";
static constexpr const char *EnclosingFnKey = "fn";

void RedundantLookupCheck::registerMatchers(MatchFinder *Finder) {
  auto MatchesContainerNameRegex =
      matchesName(ContainerNameRegex, llvm::Regex::IgnoreCase);

  // Match that the expression is a record type with a name that contains "map"
  // or "set".
  auto RecordCalledMapOrSet =
      expr(ignoringImpCasts(hasType(hasUnqualifiedDesugaredType(recordType(
               hasDeclaration(namedDecl(MatchesContainerNameRegex)))))))
          .bind(ObjKey);

  auto SubscriptCall =
      cxxOperatorCallExpr(hasOverloadedOperatorName("[]"), argumentCountIs(2),
                          hasArgument(0, RecordCalledMapOrSet),
                          hasArgument(1, expr().bind(LookupKey)));

  auto LookupMethodCalls =
      cxxMemberCallExpr(on(RecordCalledMapOrSet), argumentCountIs(1),
                        hasArgument(0, expr().bind(LookupKey)),
                        callee(cxxMethodDecl(hasAnyName(LookupMethodNames))));

  // Match any lookup or subscript calls that are not in a macro expansion.
  auto AnyLookup = callExpr(unless(hasMacroSourceRange()),
                            anyOf(SubscriptCall, LookupMethodCalls))
                       .bind(LookupCallKey);

  // We need to collect all lookups in a function to be able to report them in
  // batches.
  Finder->addMatcher(
      functionDecl(hasBody(compoundStmt(forEachDescendant(AnyLookup))))
          .bind(EnclosingFnKey),
      this);
}

/// Hash the container object expr along with the key used for lookup and the
/// enclosing function together.
static unsigned hashLookupEvent(const ASTContext &Ctx,
                                const FunctionDecl *EnclosingFn,
                                const Expr *LookupKey,
                                const Expr *ContainerObject) {
  llvm::FoldingSetNodeID ID;
  ID.AddPointer(EnclosingFn);

  LookupKey->Profile(ID, Ctx, /*Canonical=*/true,
                     /*ProfileLambdaExpr=*/true);
  ContainerObject->Profile(ID, Ctx, /*Canonical=*/true,
                           /*ProfileLambdaExpr=*/true);
  return ID.ComputeHash();
}

void RedundantLookupCheck::check(const MatchFinder::MatchResult &Result) {
  SM = Result.SourceManager;

  const auto *EnclosingFn =
      Result.Nodes.getNodeAs<FunctionDecl>(EnclosingFnKey);
  const auto *LookupCall = Result.Nodes.getNodeAs<CallExpr>(LookupCallKey);
  const auto *Key = Result.Nodes.getNodeAs<Expr>(LookupKey);
  const auto *ContainerObject = Result.Nodes.getNodeAs<Expr>(ObjKey);

  const unsigned LookupHash =
      hashLookupEvent(*Result.Context, EnclosingFn, ContainerObject, Key);
  RegisteredLookups.try_emplace(LookupHash).first->second.insert(LookupCall);
}

void RedundantLookupCheck::onEndOfTranslationUnit() {
  auto ByBeginLoc = [this](const CallExpr *Lookup1, const CallExpr *Lookup2) {
    return SM->isBeforeInTranslationUnit(Lookup1->getBeginLoc(),
                                         Lookup2->getBeginLoc());
  };

  // Process the found lookups of each function.
  for (const auto &LookupGroup : llvm::make_second_range(RegisteredLookups)) {
    if (LookupGroup.size() < 2)
      continue;

    llvm::SmallVector<const CallExpr *> SortedGroup;
    SortedGroup.reserve(LookupGroup.size());
    llvm::append_range(SortedGroup, LookupGroup);
    llvm::sort(SortedGroup, ByBeginLoc);

    const CallExpr *FirstLookupCall = SortedGroup.front();
    diag(FirstLookupCall->getBeginLoc(), "possibly redundant container lookups")
        << FirstLookupCall->getSourceRange();

    for (const CallExpr *LookupCall : llvm::drop_begin(SortedGroup)) {
      diag(LookupCall->getBeginLoc(), "next lookup here", DiagnosticIDs::Note)
          << LookupCall->getSourceRange();
    }
  }

  RegisteredLookups.clear();
}

} // namespace clang::tidy::performance
