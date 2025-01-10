//===----- NondeterministicPointerIterationOrderCheck.cpp - clang-tidy ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NondeterministicPointerIterationOrderCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void NondeterministicPointerIterationOrderCheck::registerMatchers(
    MatchFinder *Finder) {

  auto LoopVariable = varDecl(hasType(
      qualType(hasCanonicalType(anyOf(referenceType(), pointerType())))));

  auto RangeInit = declRefExpr(to(varDecl(
      hasType(recordDecl(hasAnyName("std::unordered_set", "std::unordered_map",
                                    "std::unordered_multiset",
                                    "std::unordered_multimap"))
                  .bind("recorddecl")))));

  Finder->addMatcher(cxxForRangeStmt(hasLoopVariable(LoopVariable),
                                     hasRangeInit(RangeInit.bind("rangeinit")))
                         .bind("cxxForRangeStmt"),
                     this);

  auto SortFuncM = callee(functionDecl(hasAnyName(
      "std::is_sorted", "std::nth_element", "std::sort", "std::partial_sort",
      "std::partition", "std::stable_partition", "std::stable_sort")));

  auto IteratesPointerEltsM = hasArgument(
      0,
      cxxMemberCallExpr(on(hasType(cxxRecordDecl(has(fieldDecl(hasType(qualType(
          hasCanonicalType(pointsTo(hasCanonicalType(pointerType()))))))))))));

  Finder->addMatcher(
      callExpr(allOf(SortFuncM, IteratesPointerEltsM)).bind("sortsemantic"),
      this);
}

void NondeterministicPointerIterationOrderCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *ForRangePointers =
      Result.Nodes.getNodeAs<CXXForRangeStmt>("cxxForRangeStmt");

  if ((ForRangePointers) && !(ForRangePointers->getBeginLoc().isMacroID())) {
    const auto *RangeInit = Result.Nodes.getNodeAs<Stmt>("rangeinit");
    if (const auto *ClassTemplate =
            Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>(
                "recorddecl")) {
      const TemplateArgumentList &TemplateArgs =
          ClassTemplate->getTemplateArgs();
      const bool IsAlgoArgPointer =
          TemplateArgs[0].getAsType()->isPointerType();

      if (IsAlgoArgPointer) {
        SourceRange R = RangeInit->getSourceRange();
        diag(R.getBegin(), "iteration of pointers is nondeterministic") << R;
      }
    }
    return;
  }
  const auto *SortPointers = Result.Nodes.getNodeAs<Stmt>("sortsemantic");

  if ((SortPointers) && !(SortPointers->getBeginLoc().isMacroID())) {
    SourceRange R = SortPointers->getSourceRange();
    diag(R.getBegin(), "sorting pointers is nondeterministic") << R;
  }
}

} // namespace clang::tidy::bugprone
