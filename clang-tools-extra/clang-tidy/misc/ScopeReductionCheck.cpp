//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This checker uses a 9-step algorithm to accomplish scope analysis of a
// variable and determine if it can be declared in a smaller scope. Note that
// the clang-tidy framework is aimed mainly at supporting text-manipulation,
// diagnostics, or common AST patterns. Scope reduction analysis is
// quite specialized, and there's not much support specifically for
// those steps. Perhaps someone else knows better and can help simplify
// this code in a more concrete way other than simply suggesting it can
// be simpler.
//
// The 9-step algorithm used by this checker for scope reduction analysis is:
// 1) AST Matcher Filtering
//    - Only match variables within functions (hasAncestor(functionDecl())
//    - Exclude for-loop declared variables
//       (unless(hasParent(declStmt(hasParent(forStmt))))))
//    - Exclude variables with function call initializers
//       (unless(hasInitializer(...)))
//    - Exclude parameters from analysis
//       (unless(parmVarDecl())
//    - Exclude try-catch variables (unless(hasParent(cxxCatchStmt())))
// 2) Collect variable uses
//    - Find all DeclRefExpr nodes that reference the variable
// 3) Build scope chains
//    - For each use, find all compound statements that contain it (from
//      innermost to outermost)
// 4) Find the innermost compound statement that contains all uses
//    - This is the smallest scope where the variable could be declared
// 5) Check for modification detection
//    - Skip analysis for initialized variables modified with unary operators
//    - Skip analysis for initialized variables moved into switch bodies
//    - This prevents moving variables where initialization would be lost or
//    conditional
// 6) Check for loop body placement
//    - Skip analysis if suggested scope would place variable inside loop body
//    - This prevents suggesting moving variables into loop bodies (inefficient)
// 7) Switch case analysis
//    - Check if variable uses span multiple case labels in the same switch
//    - Skip analysis if so, as variables cannot be declared in switch body
// 8) Verify scope nesting and report
//    - Find the compound statement containing the variable declaration
//    - Only report if the usage scope is nested within the declaration scope
//    - This ensures we only suggest moving variables to smaller scopes
// 9) Alternative analysis - check for for-loop initialization opportunity
//    - Only runs if compound statement analysis didn't find a smaller scope
//    - Only check local variables, not parameters
//    - Determine if all uses are within the same for-loop and suggest
//      for-loop initialization, but only if for-loop is in smaller scope
//
// The algorithm works by finding the smallest scope that could contain the
// variable declaration while still encompassing all its uses, but only reports
// when that scope is smaller than the current declaration scope.

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
              unless(hasParent(declStmt(hasParent(cxxForRangeStmt())))),
              unless(hasParent(cxxCatchStmt())),
              unless(hasInitializer(anyOf(
                  hasDescendant(callExpr()), hasDescendant(cxxMemberCallExpr()),
                  hasDescendant(cxxOperatorCallExpr()), callExpr(),
                  cxxMemberCallExpr(), cxxOperatorCallExpr()))))
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

  // Step 5: Check for modification detection
  // Skip analysis for initialized variables that would lose initialization
  // semantics or make initialization conditional
  if (InnermostScope) {
    for (const auto *Use : Uses) {
      // Check if initialized variable is modified with unary operators
      // Moving would lose initialization value
      if (Var->hasInit()) {
        auto UseParents =
            Result.Context->getParentMapContext().getParents(*Use);
        if (!UseParents.empty()) {
          if (const auto *UnaryOp = UseParents[0].get<UnaryOperator>()) {
            if (UnaryOp->isIncrementDecrementOp()) {
              return; // Skip - moving initialized variable that gets
                      // incremented/decremented would lose initialization
            }
          }
        }
      }

      // Check if initialized variable would be moved into a switch body
      // Moving would make initialization conditional on case execution
      if (Var->hasInit() && InnermostScope) {
        auto ScopeParents =
            Result.Context->getParentMapContext().getParents(*InnermostScope);
        if (!ScopeParents.empty()) {
          const Stmt *ScopeParent = ScopeParents[0].get<Stmt>();
          if (!ScopeParent) {
            if (const auto *DeclParent = ScopeParents[0].get<Decl>()) {
              auto DeclParentNodes =
                  Result.Context->getParentMapContext().getParents(*DeclParent);
              if (!DeclParentNodes.empty())
                ScopeParent = DeclParentNodes[0].get<Stmt>();
            }
          }

          // Check if the suggested scope is the body of a switch statement
          if (const auto *Switch = dyn_cast_or_null<SwitchStmt>(ScopeParent)) {
            if (Switch->getBody() == InnermostScope) {
              return; // Skip - moving initialized variable into switch body
                      // would make initialization conditional
            }
          }
        }
      }

      // Step 6: Check if this use is inside a loop
      auto &Parents = Result.Context->getParentMapContext();

      // Check if use is in any type of loop and get the specific loop
      const Stmt *ContainingLoop = nullptr;
      if (const auto *ForLoop = findAncestorOfType<ForStmt>(Use, Parents)) {
        ContainingLoop = ForLoop;
      } else if (const auto *WhileLoop =
                     findAncestorOfType<WhileStmt>(Use, Parents)) {
        ContainingLoop = WhileLoop;
      } else if (const auto *DoLoop =
                     findAncestorOfType<DoStmt>(Use, Parents)) {
        ContainingLoop = DoLoop;
      } else if (const auto *RangeLoop =
                     findAncestorOfType<CXXForRangeStmt>(Use, Parents)) {
        ContainingLoop = RangeLoop;
      }

      // If use is in a loop, check if suggested scope is inside that loop
      if (ContainingLoop) {
        const Stmt *CheckScope = InnermostScope;
        bool ScopeInsideLoop = false;

        while (CheckScope) {
          if (CheckScope == ContainingLoop) {
            ScopeInsideLoop = true;
            break;
          }
          auto CheckParents =
              Result.Context->getParentMapContext().getParents(*CheckScope);
          if (CheckParents.empty())
            break;
          CheckScope = CheckParents[0].get<Stmt>();
        }

        if (ScopeInsideLoop)
          return; // Skip if suggested scope is inside loop body
      }
    }
  }

  // Step 7: Check if current variable declaration can be moved to a smaller
  // scope
  if (InnermostScope) {
    // Check if variable uses span multiple case labels in the same switch
    // If so, the only common scope would be the switch body, which is invalid
    // for declarations
    std::set<const SwitchCase *> CaseLabels;
    bool UsesInSwitch = false;

    for (const auto *Use : Uses) {
      // Find containing switch case using helper
      const SwitchCase *ContainingCase = findContainingSwitchCase(Use, Parents);

      if (ContainingCase) {
        CaseLabels.insert(ContainingCase);
        UsesInSwitch = true;
      }
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

    // Step 8: Verify that usage scope is nested within declaration scope
    // Only report if we can move the variable to a smaller scope
    if (VarScope && VarScope != InnermostScope) {
      // Walk up from innermost usage scope to see if declaration scope is
      // reached
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

      // Only report if the usage scope is truly nested within the declaration
      // scope
      if (IsNested) {
        diag(Var->getLocation(),
             "variable '%0' can be declared in a smaller scope")
            << Var->getName();

        emitUsageNotes(Uses);

        diag(InnermostScope->getBeginLoc(), "can be declared in this scope",
             DiagnosticIDs::Note);
        return;
      }
    }
  }

  // Step 9: Alternative analysis - check for for-loop initialization
  // opportunity This only runs if the compound statement analysis didn't find
  // a smaller scope Only check local variables, not parameters
  const ForStmt *CommonForLoop = nullptr;
  bool AllUsesInSameForLoop = true;

  for (const auto *Use : Uses) {
    // Find containing for-loop using helper
    const ForStmt *ContainingForLoop =
        findAncestorOfType<ForStmt>(Use, Parents);

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
    // Check if for-loop scope is broader than current declaration scope
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

    // Only report if for-loop is in a smaller scope than current declaration
    if (VarScope) {
      const Stmt *CheckScope = CommonForLoop;
      bool IsSmaller = false;

      while (CheckScope) {
        if (CheckScope == VarScope) {
          IsSmaller = true;
          break;
        }
        auto CheckParents = Parents.getParents(*CheckScope);
        if (CheckParents.empty())
          break;
        CheckScope = CheckParents[0].get<Stmt>();
      }

      if (IsSmaller) {
        // Check if variable is modified in this for-loop's increment
        // Moving to for-loop initialization would create complex, hard-to-read
        // code
        if (CommonForLoop && CommonForLoop->getInc()) {
          for (const auto *Use : Uses) {
            auto UseParents =
                Result.Context->getParentMapContext().getParents(*Use);
            if (!UseParents.empty()) {
              if (const auto *BinOp = UseParents[0].get<BinaryOperator>()) {
                if (BinOp->isAssignmentOp() && BinOp->getLHS() == Use) {
                  if (BinOp == CommonForLoop->getInc())
                    return; // Skip - variable modified in for-loop increment
                }
              }
            }
          }
        }

        // Check if variable's address is taken in the same for-loop
        // Moving would break the dependency
        if (CommonForLoop) {
          for (const auto *Use : Uses) {
            auto UseParents =
                Result.Context->getParentMapContext().getParents(*Use);
            if (!UseParents.empty()) {
              if (const auto *UnaryOp = UseParents[0].get<UnaryOperator>()) {
                if (UnaryOp->getOpcode() == UO_AddrOf) {
                  // Check if this address-of is in the for-loop init
                  if (CommonForLoop->getInit()) {
                    const Stmt *Current = UnaryOp;
                    while (Current) {
                      if (Current == CommonForLoop->getInit()) {
                        return; // Skip - variable's address taken in for-loop
                                // init
                      }
                      auto CurrentParents =
                          Result.Context->getParentMapContext().getParents(
                              *Current);
                      if (CurrentParents.empty())
                        break;
                      Current = CurrentParents[0].get<Stmt>();
                    }
                  }
                }
              }
            }
          }
        }

        diag(Var->getLocation(),
             "variable '%0' can be declared in for-loop initialization")
            << Var->getName();

        // Skip usage notes for for-loops - usage pattern is obvious
        diag(CommonForLoop->getBeginLoc(), "can be declared in this for-loop",
             DiagnosticIDs::Note);
      }
    }
  }
}

void ScopeReductionCheck::emitUsageNotes(
    const llvm::SmallVector<const DeclRefExpr *, 8> &Uses) {
  const size_t MaxUsageNotes = 3;
  for (const auto *Use : take(Uses, MaxUsageNotes))
    diag(Use->getLocation(), "used here", DiagnosticIDs::Note);
  if (Uses.size() > MaxUsageNotes) {
    diag(Uses[MaxUsageNotes]->getLocation(), "and %0 more uses...",
         DiagnosticIDs::Note)
        << (Uses.size() - MaxUsageNotes);
  }
}

} // namespace clang::tidy::misc

template <typename T>
const T *clang::tidy::misc::ScopeReductionCheck::findAncestorOfType(
    const Stmt *Start, ParentMapContext &Parents) {
  const Stmt *Current = Start;
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

    if (const auto *Result = dyn_cast<T>(Parent))
      return Result;
    Current = Parent;
  }
  return nullptr;
}

bool clang::tidy::misc::ScopeReductionCheck::isInLoop(
    const Stmt *Use, ParentMapContext &Parents) {
  return findAncestorOfType<ForStmt>(Use, Parents) ||
         findAncestorOfType<WhileStmt>(Use, Parents) ||
         findAncestorOfType<DoStmt>(Use, Parents) ||
         findAncestorOfType<CXXForRangeStmt>(Use, Parents);
}

const clang::SwitchCase *
clang::tidy::misc::ScopeReductionCheck::findContainingSwitchCase(
    const Stmt *Use, ParentMapContext &Parents) {
  return findAncestorOfType<clang::SwitchCase>(Use, Parents);
}
