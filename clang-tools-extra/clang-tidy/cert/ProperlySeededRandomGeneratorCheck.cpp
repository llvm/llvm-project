//===--- ProperlySeededRandomGeneratorCheck.cpp - clang-tidy---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProperlySeededRandomGeneratorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

ProperlySeededRandomGeneratorCheck::ProperlySeededRandomGeneratorCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawDisallowedSeedTypes(
          Options.get("DisallowedSeedTypes", "time_t,std::time_t")) {
  StringRef(RawDisallowedSeedTypes).split(DisallowedSeedTypes, ',');
}

void ProperlySeededRandomGeneratorCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "DisallowedSeedTypes", RawDisallowedSeedTypes);
}

void ProperlySeededRandomGeneratorCheck::registerMatchers(MatchFinder *Finder) {
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
      cxxConstructExpr(RandomGeneratorEngineTypeMatcher).bind("ctor"), this);

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

void ProperlySeededRandomGeneratorCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  if (Ctor)
    checkSeed(Result, Ctor);

  const auto *Func = Result.Nodes.getNodeAs<CXXMemberCallExpr>("seed");
  if (Func)
    checkSeed(Result, Func);

  const auto *Srand = Result.Nodes.getNodeAs<CallExpr>("srand");
  if (Srand)
    checkSeed(Result, Srand);
}

template <class T>
void ProperlySeededRandomGeneratorCheck::checkSeed(
    const MatchFinder::MatchResult &Result, const T *Func) {
  if (Func->getNumArgs() == 0 || Func->getArg(0)->isDefaultArgument()) {
    diag(Func->getExprLoc(),
         "random number generator seeded with a default argument will generate "
         "a predictable sequence of values");
    return;
  }

  llvm::APSInt Value;
  if (Func->getArg(0)->EvaluateAsInt(Value, *Result.Context)) {
    diag(Func->getExprLoc(),
         "random number generator seeded with a constant value will generate a "
         "predictable sequence of values");
    return;
  }

  const std::string SeedType(
      Func->getArg(0)->IgnoreCasts()->getType().getAsString());
  if (llvm::find(DisallowedSeedTypes, SeedType) != DisallowedSeedTypes.end()) {
    diag(Func->getExprLoc(),
         "random number generator seeded with a disallowed source of seed "
         "value will generate a predictable sequence of values");
    return;
  }
}

} // namespace cert
} // namespace tidy
} // namespace clang
