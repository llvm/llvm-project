//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RandomGeneratorSeedCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
AST_MATCHER_P(CXXConstructExpr, hasImplicitCtorInitField,
              ast_matchers::internal::Matcher<Decl>, InnerMatcher) {
  const DynTypedNodeList Parents =
      Finder->getASTContext().getParentMapContext().getParents(Node);
  if (Parents.empty())
    return false;
  if (const auto *Ctor = Parents[0].get<CXXConstructorDecl>()) {
    for (const CXXCtorInitializer *Init : Ctor->inits())
      if (!Init->isWritten() && Init->getInit() == &Node && Init->getMember())
        return InnerMatcher.matches(*Init->getMember(), Finder, Builder);
  }
  return false;
}
} // namespace

RandomGeneratorSeedCheck::RandomGeneratorSeedCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawDisallowedSeedTypes(
          Options.get("DisallowedSeedTypes", "time_t,std::time_t")) {
  RawDisallowedSeedTypes.split(DisallowedSeedTypes, ',');
}

void RandomGeneratorSeedCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "DisallowedSeedTypes", RawDisallowedSeedTypes);
}

void RandomGeneratorSeedCheck::registerMatchers(MatchFinder *Finder) {
  auto RandomGeneratorEngineDecl = cxxRecordDecl(hasAnyName(
      "::std::linear_congruential_engine", "::std::mersenne_twister_engine",
      "::std::subtract_with_carry_engine", "::std::discard_block_engine",
      "::std::independent_bits_engine", "::std::shuffle_order_engine"));
  auto RandomGeneratorEngineTypeMatcher = hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(RandomGeneratorEngineDecl))));

  // std::mt19937 engine;
  // engine.seed();
  //        ^
  // engine.seed(1);
  //        ^
  // const int x = 1;
  // engine.seed(x);
  //        ^
  Finder->addMatcher(
      cxxMemberCallExpr(
          has(memberExpr(has(declRefExpr(RandomGeneratorEngineTypeMatcher)),
                         member(hasName("seed")),
                         unless(hasDescendant(cxxThisExpr())))))
          .bind("seed"),
      this);

  // std::mt19937 engine;
  //              ^
  // std::mt19937 engine(1);
  //              ^
  // const int x = 1;
  // std::mt19937 engine(x);
  //              ^
  Finder->addMatcher(
      traverse(TK_AsIs, cxxConstructExpr(RandomGeneratorEngineTypeMatcher,
                                         optionally(hasImplicitCtorInitField(
                                             fieldDecl().bind("field"))))
                            .bind("ctor")),
      this);

  // srand();
  // ^
  // const int x = 1;
  // srand(x);
  // ^
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName("::srand", "::std::srand"))))
          .bind("srand"),
      this);
}

void RandomGeneratorSeedCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  if (Ctor)
    checkSeed(Result, Ctor, Result.Nodes.getNodeAs<FieldDecl>("field"));

  const auto *Func = Result.Nodes.getNodeAs<CXXMemberCallExpr>("seed");
  if (Func)
    checkSeed(Result, Func);

  const auto *Srand = Result.Nodes.getNodeAs<CallExpr>("srand");
  if (Srand)
    checkSeed(Result, Srand);
}

template <class T>
void RandomGeneratorSeedCheck::checkSeed(const MatchFinder::MatchResult &Result,
                                         const T *Func,
                                         const FieldDecl *Field) {
  if (Func->getNumArgs() == 0 || Func->getArg(0)->isDefaultArgument()) {
    diag(Func->getExprLoc(),
         "random number generator seeded with a default argument will generate "
         "a predictable sequence of values");
    if (Field)
      diag(Field->getLocation(),
           "field %0 is implicitly initialized with a default seed argument",
           DiagnosticIDs::Note)
          << Field;

    return;
  }

  Expr::EvalResult EVResult;
  if (Func->getArg(0)->EvaluateAsInt(EVResult, *Result.Context)) {
    diag(Func->getExprLoc(),
         "random number generator seeded with a constant value will generate a "
         "predictable sequence of values");
    return;
  }

  const std::string SeedType(
      Func->getArg(0)->IgnoreCasts()->getType().getAsString());
  if (llvm::is_contained(DisallowedSeedTypes, SeedType)) {
    diag(Func->getExprLoc(),
         "random number generator seeded with a disallowed source of seed "
         "value will generate a predictable sequence of values");
    return;
  }
}

} // namespace clang::tidy::bugprone
