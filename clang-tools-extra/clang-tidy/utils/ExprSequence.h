//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_EXPRSEQUENCE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_EXPRSEQUENCE_H

#include "clang/Analysis/CFG.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "../ClangTidy.h"

namespace clang::tidy::utils {

/// Provides information about the evaluation order of (sub-)expressions within
/// a `CFGBlock`.
///
/// While a `CFGBlock` does contain individual `CFGElement`s for some
/// sub-expressions, the order in which those `CFGElement`s appear reflects
/// only one possible order in which the sub-expressions may be evaluated.
/// However, we want to warn if any of the potential evaluation orders can lead
/// to a use-after-move, not just the one contained in the `CFGBlock`.
///
/// This class implements only a simplified version of the C++ sequencing
/// rules. The main limitation is that we do not distinguish between value
/// computation and side effect -- see the "Implementation" section for more
/// details.
///
/// Note: `SequenceChecker` from SemaChecking.cpp does a similar job (and much
/// more thoroughly), but using it would require
/// - Pulling `SequenceChecker` out into a header file (i.e. making it part of
///   the API),
/// - Removing the dependency of `SequenceChecker` on `Sema`, and
/// - (Probably) modifying `SequenceChecker` to make it suitable to be used in
///   this context.
/// For the moment, it seems preferable to re-implement our own version of
/// sequence checking that is special-cased to what we need here.
///
/// Implementation
/// --------------
///
/// `ExprSequence` uses two types of sequencing edges between nodes in the AST:
///
/// - Every `Stmt` is assumed to be sequenced after its children. This is
///   overly optimistic because the standard only states that value computations
///   of operands are sequenced before the value computation of the operator,
///   making no guarantees about side effects (in general).
///
///   For our purposes, this rule is sufficient, however, because this check is
///   interested in operations on objects, which are generally performed through
///   function calls (whether explicit and implicit). Function calls guarantee
///   that the value computations and side effects for all function arguments
///   are sequenced before the execution of the function.
///
/// - In addition, some `Stmt`s are known to be sequenced before or after
///   their siblings. For example, the `Stmt`s that make up a `CompoundStmt`are
///   all sequenced relative to each other. The function
///   `getSequenceSuccessor()` implements these sequencing rules.
class ExprSequence {
public:
  /// Initializes this `ExprSequence` with sequence information for the given
  /// `CFG`. `Root` is the root statement the CFG was built from.
  ExprSequence(const CFG *TheCFG, const Stmt *Root, ASTContext *TheContext);

  /// Returns whether \p Before is sequenced before \p After.
  bool inSequence(const Stmt *Before, const Stmt *After) const;

  /// Returns whether \p After can potentially be evaluated after \p Before.
  /// This is exactly equivalent to `!inSequence(After, Before)` but makes some
  /// conditions read more naturally.
  bool potentiallyAfter(const Stmt *After, const Stmt *Before) const;

private:
  // Returns the sibling of \p S (if any) that is directly sequenced after \p S,
  // or nullptr if no such sibling exists. For example, if \p S is the child of
  // a `CompoundStmt`, this would return the Stmt that directly follows \p S in
  // the `CompoundStmt`.
  //
  // As the sequencing of many constructs that change control flow is already
  // encoded in the `CFG`, this function only implements the sequencing rules
  // for those constructs where sequencing cannot be inferred from the `CFG`.
  const Stmt *getSequenceSuccessor(const Stmt *S) const;

  const Stmt *resolveSyntheticStmt(const Stmt *S) const;

  ASTContext *Context;
  const Stmt *Root;

  llvm::DenseMap<const Stmt *, const Stmt *> SyntheticStmtSourceMap;
};

/// Maps `Stmt`s to the `CFGBlock` that contains them. Some `Stmt`s may be
/// contained in more than one `CFGBlock`; in this case, they are mapped to the
/// innermost block (i.e. the one that is furthest from the root of the tree).
class StmtToBlockMap {
public:
  /// Initializes the map for the given `CFG`.
  StmtToBlockMap(const CFG *TheCFG, ASTContext *TheContext);

  /// Returns the block that \p S is contained in. Some `Stmt`s may be contained
  /// in more than one `CFGBlock`; in this case, this function returns the
  /// innermost block (i.e. the one that is furthest from the root of the tree).
  const CFGBlock *blockContainingStmt(const Stmt *S) const;

private:
  ASTContext *Context;

  llvm::DenseMap<const Stmt *, const CFGBlock *> Map;
};

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_EXPRSEQUENCE_H
