//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This checker uses a 7-step algorithm to accomplish scope analysis of a
// variable and determine if it's in too large a scope. Note that the
// clang-tidy framework is aimed mainly at supporting text-manipulation,
// diagnostics, or common AST patterns. Scope reduction analysis is
// quite specialized, and there's not much support specifically for
// those steps. Perhaps someone else knows better and can help simplify
// this code in a more concrete way other than simply suggesting it can
// be simpler.
//
// The 7-step algorithm used by this checker for scope reduction analysis is:
// 1) AST Matcher Filtering
//    - Only match variables within functions (hasAncestor(functionDecl())
//    - Exclude for-loop declared variables
//       (unless(hasParent(declStmt(hasParent(forStmt))))))
//    - Exclude variables with function call initializors
//       (unless(hasInitializer(...)))
//    - Exclude parameters from analysis
//       (unless(parmVarDecl())
// 2) Collect variable uses
//    - find all DeclRefExpr nodes that reference the variable
// 3) Build scope chains
//    - for each use, find all compound statements that contain it (from
//      innermost to outermost)
// 4) Find the innermost compound statement that contains all uses
//    - This is the smallest scope where the variable could be declared
// 5) Find declaration scope
//    - Locate the compound statement containing the variable declaration
// 6) Verify nesting
//    - Ensure the usage scope is actually nested within the declaration scope
// 7) Alternate analysis - check for for-loop initialization opportunity
//    - This is only run if compound stmt analysis didn't find smaller scope
//    - Only check local variables, not parameters
//    - Determine if all uses are within the same for-loop and suggest
//      for-loop initialization
//
// The algo works by finding the smallest scope that could contain the variable
// declaration while still encompassing all it's uses.

#include "ScopeReductionCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

static void
collectVariableUses(const clang::Stmt *S, const clang::VarDecl *Var,
                    llvm::SmallVector<const clang::DeclRefExpr *, 8> &Uses) {
  if (!S || !Var)
    return;

  const llvm::SmallPtrSet<const clang::DeclRefExpr *, 16> DREs =
      clang::tidy::utils::decl_ref_expr::allDeclRefExprs(*Var, *S,
                                                         Var->getASTContext());

  // Copy the results into the provided SmallVector
  Uses.clear();
  Uses.append(DREs.begin(), DREs.end());
}

void ScopeReductionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      varDecl(hasLocalStorage(), unless(hasGlobalStorage()),
              hasAncestor(functionDecl()), unless(parmVarDecl()),
              unless(hasParent(declStmt(hasParent(forStmt())))),
              unless(hasInitializer(anyOf(callExpr(), cxxMemberCallExpr(),
                                          cxxOperatorCallExpr()))))
          .bind("var"),
      this);
}

void ScopeReductionCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var");

  assert(Var);

  // Step 1: Filter out variables declared in for-loop initializations
  // These variables are already in their optimal scope and shouldn't be
  // analyzed
  auto &Parents = Result.Context->getParentMapContext();

  const auto *Function = dyn_cast<FunctionDecl>(Var->getDeclContext());
  assert(Function);

  // Step 2: Collect all uses of this variable within the function
  llvm::SmallVector<const DeclRefExpr *, 8> Uses;
  collectVariableUses(Function->getBody(), Var, Uses);

  // No uses, return with no diagnostics
  if (Uses.empty())
    return;

  // Step 3: For each variable use, find all compound statements that contain it
  // This builds a "scope chain" from innermost to outermost for each use
  const CompoundStmt *InnermostScope = nullptr;

  // For each use, find all compound statements that contain it
  llvm::SmallVector<llvm::SmallVector<const CompoundStmt *, 4>, 8>
      UseScopeChains;

  for (const auto *Use : Uses) {
    llvm::SmallVector<const CompoundStmt *, 4> ScopeChain;
    const Stmt *Current = Use;

    // Walk up the AST from this use to fins all containing compound stmts
    while (Current) {
      auto ParentNodes = Parents.getParents(*Current);
      if (ParentNodes.empty())
        break;

      const Stmt *Parent = ParentNodes[0].get<Stmt>();
      if (!Parent) {
        // Try to get Decl parent and continue from there
        if (const auto *DeclParent = ParentNodes[0].get<Decl>()) {
          auto DeclParentNodes = Parents.getParents(*DeclParent);
          if (!DeclParentNodes.empty())
            Parent = DeclParentNodes[0].get<Stmt>();
        }
        if (!Parent)
          break;
      }

      if (const auto *CS = dyn_cast<CompoundStmt>(Parent))
        ScopeChain.push_back(CS);

      Current = Parent;
    }

    if (!ScopeChain.empty())
      UseScopeChains.push_back(ScopeChain);
  }

  // Step 4: Find the innermost scope that contains all uses
  //         This is the smallest scope where var could be declared
  if (!UseScopeChains.empty()) {
    // Start with first use's innermost scope
    InnermostScope = UseScopeChains[0][0];

    // For each subsequent use, find common ancestor scope
    for (const auto &ScopeChain : llvm::drop_begin(UseScopeChains)) {
      const CompoundStmt *CommonScope = nullptr;

      // Find first scope that appears in both chains (common ancestor)
      for (const auto *Scope1 : UseScopeChains[0]) {
        for (const auto *Scope2 : ScopeChain) {
          if (Scope1 == Scope2) {
            CommonScope = Scope1;
            break;
          }
        }
        if (CommonScope)
          break;
      }

      if (CommonScope)
        InnermostScope = CommonScope;
    }
  }

  // Step 5: Check if current var declaration is broader than necessary
  if (InnermostScope) {
    // Check if variable uses span multiple case labels in the same switch
    // If so, the only common scope would be the switch body, which is invalid
    // for declarations
    std::set<const SwitchCase *> CaseLabels;
    bool UsesInSwitch = false;

    for (const auto *Use : Uses) {
      const Stmt *Current = Use;
      const SwitchCase *ContainingCase = nullptr;

      // Walk up to find containing case label
      while (Current) {
        auto ParentNodes = Parents.getParents(*Current);
        if (ParentNodes.empty())
          break;

        const Stmt *Parent = ParentNodes[0].get<Stmt>();
        if (!Parent)
          break;

        if (const auto *CaseStmt = dyn_cast<SwitchCase>(Parent)) {
          ContainingCase = CaseStmt;
          UsesInSwitch = true;
          break;
        }
        Current = Parent;
      }

      if (ContainingCase)
        CaseLabels.insert(ContainingCase);
    }

    // If uses span multiple case labels, skip analysis
    if (UsesInSwitch && CaseLabels.size() > 1) {
      return; // Cannot declare variables in switch body when used across
              // multiple cases
    }

    // Find the compound statement containing the variable declaration
    const DynTypedNode Current = DynTypedNode::create(*Var);
    const CompoundStmt *VarScope = nullptr;

    auto ParentNodes = Parents.getParents(Current);
    while (!ParentNodes.empty()) {
      const Stmt *Parent = ParentNodes[0].get<Stmt>();
      if (!Parent)
        break;

      if (const auto *CS = dyn_cast<CompoundStmt>(Parent)) {
        VarScope = CS;
        break;
      }
      ParentNodes = Parents.getParents(*Parent);
    }

    // Step 6: Verify that usage scope is nested within decl scope
    if (VarScope && VarScope != InnermostScope) {
      // Walk up from innermost usage to see if the decl scope is reached
      const Stmt *CheckScope = InnermostScope;
      bool IsNested = false;

      while (CheckScope) {
        auto CheckParents = Parents.getParents(*CheckScope);
        if (CheckParents.empty())
          break;

        const Stmt *CheckParent = CheckParents[0].get<Stmt>();
        if (CheckParent == VarScope) {
          IsNested = true;
          break;
        }
        CheckScope = CheckParent;
      }

      // Only report if the usage scope is truly nested within the decl scope
      if (IsNested) {
        diag(Var->getLocation(),
             "variable '%0' can be declared in a smaller scope")
            << Var->getName();
        return;
      }
    }
  }

  // Step 7: Alternative analysis - check for for-loop initialization
  // opportunity This only runs if the compound statement analysis didn't find a
  // smaller scope Only check local variables, not parameters
  const ForStmt *CommonForLoop = nullptr;
  bool AllUsesInSameForLoop = true;

  for (const auto *Use : Uses) {
    const ForStmt *ContainingForLoop = nullptr;
    const Stmt *Current = Use;

    // Walk up the AST to find a containing ForStmt
    while (Current) {
      auto ParentNodes = Parents.getParents(*Current);
      if (ParentNodes.empty())
        break;

      if (const auto *FS = ParentNodes[0].get<ForStmt>()) {
        ContainingForLoop = FS;
        break;
      }

      const Stmt *Parent = ParentNodes[0].get<Stmt>();
      if (!Parent) {
        // Handle Decl parents like we do in the existing logic
        if (const auto *DeclParent = ParentNodes[0].get<Decl>()) {
          auto DeclParentNodes = Parents.getParents(*DeclParent);
          if (!DeclParentNodes.empty())
            Parent = DeclParentNodes[0].get<Stmt>();
        }
        if (!Parent)
          break;
      }
      Current = Parent;
    }

    if (!ContainingForLoop) {
      AllUsesInSameForLoop = false;
      break;
    }

    if (!CommonForLoop) {
      CommonForLoop = ContainingForLoop;
    } else if (CommonForLoop != ContainingForLoop) {
      AllUsesInSameForLoop = false;
      break;
    }
  }

  if (AllUsesInSameForLoop && CommonForLoop) {
    diag(Var->getLocation(),
         "variable '%0' can be declared in for-loop initialization")
        << Var->getName();
    return;
  }
}

} // namespace clang::tidy::misc
