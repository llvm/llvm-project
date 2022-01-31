//===-- SimplifyBooleanExprMatchers.h - clang-tidy ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"

namespace clang {
namespace ast_matchers {

/// Matches the substatement associated with a case, default or label statement.
///
/// Given
/// \code
///   switch (1) { case 1: break; case 2: return; break; default: return; break;
///   }
///   foo: return;
///   bar: break;
/// \endcode
///
/// caseStmt(hasSubstatement(returnStmt()))
///   matches "case 2: return;"
/// defaultStmt(hasSubstatement(returnStmt()))
///   matches "default: return;"
/// labelStmt(hasSubstatement(breakStmt()))
///   matches "bar: break;"
AST_POLYMORPHIC_MATCHER_P(hasSubstatement,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(CaseStmt, DefaultStmt,
                                                          LabelStmt),
                          internal::Matcher<Stmt>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getSubStmt(), Finder, Builder);
}

/// Matches two consecutive statements within a compound statement.
///
/// Given
/// \code
///   { if (x > 0) return true; return false; }
/// \endcode
/// compoundStmt(hasSubstatementSequence(ifStmt(), returnStmt()))
///   matches '{ if (x > 0) return true; return false; }'
AST_POLYMORPHIC_MATCHER_P2(hasSubstatementSequence,
                           AST_POLYMORPHIC_SUPPORTED_TYPES(CompoundStmt,
                                                           StmtExpr),
                           internal::Matcher<Stmt>, InnerMatcher1,
                           internal::Matcher<Stmt>, InnerMatcher2) {
  if (const CompoundStmt *CS = CompoundStmtMatcher<NodeType>::get(Node)) {
    auto It = matchesFirstInPointerRange(InnerMatcher1, CS->body_begin(),
                                         CS->body_end(), Finder, Builder);
    while (It != CS->body_end()) {
      ++It;
      if (It == CS->body_end())
        return false;
      if (InnerMatcher2.matches(**It, Finder, Builder))
        return true;
      It = matchesFirstInPointerRange(InnerMatcher1, It, CS->body_end(), Finder,
                                      Builder);
    }
  }
  return false;
}

} // namespace ast_matchers
} // namespace clang
