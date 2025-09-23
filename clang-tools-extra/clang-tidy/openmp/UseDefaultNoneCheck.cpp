//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseDefaultNoneCheck.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::openmp {

void UseDefaultNoneCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      ompExecutableDirective(
          isAllowedToContainClauseKind(llvm::omp::OMPC_default),
          anyOf(unless(hasAnyClause(ompDefaultClause())),
                hasAnyClause(
                    ompDefaultClause(unless(isNoneKind())).bind("clause"))))
          .bind("directive"),
      this);
}

void UseDefaultNoneCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Directive =
      Result.Nodes.getNodeAs<OMPExecutableDirective>("directive");
  assert(Directive != nullptr && "Expected to match some directive.");

  if (const auto *Clause = Result.Nodes.getNodeAs<OMPDefaultClause>("clause")) {
    diag(Directive->getBeginLoc(),
         "OpenMP directive '%0' specifies 'default(%1)' clause, consider using "
         "'default(none)' clause instead")
        << getOpenMPDirectiveName(Directive->getDirectiveKind())
        << getOpenMPSimpleClauseTypeName(Clause->getClauseKind(),
                                         unsigned(Clause->getDefaultKind()));
    diag(Clause->getBeginLoc(), "existing 'default' clause specified here",
         DiagnosticIDs::Note);
    return;
  }

  diag(Directive->getBeginLoc(),
       "OpenMP directive '%0' does not specify 'default' clause, consider "
       "specifying 'default(none)' clause")
      << getOpenMPDirectiveName(Directive->getDirectiveKind());
}

} // namespace clang::tidy::openmp
