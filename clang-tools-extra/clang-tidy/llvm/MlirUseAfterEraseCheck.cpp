//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MlirUseAfterEraseCheck.h"
#include "../utils/ExprSequence.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

using namespace clang::ast_matchers;
using namespace clang::tidy::utils;

namespace clang::tidy::llvm_check {

namespace {

/// Contains information about a use-after-erase.
struct UseAfterErase {
  // The DeclRefExpr that constituted the use of the operation.
  const DeclRefExpr *DeclRef;

  // Is the order in which the erase and the use are evaluated undefined?
  bool EvaluationOrderUndefined = false;

  // Does the use happen in a later loop iteration than the erase?
  //
  // We default to false and change it to true if required in find().
  bool UseHappensInLaterLoopIteration = false;
};

/// Finds uses of an `mlir::Operation` variable that are reachable after the
/// operation was erased (and maintains state required by the various internal
/// helper functions).
class UseAfterEraseFinder {
public:
  UseAfterEraseFinder(ASTContext *TheContext);

  // Within the given code block, finds the first use of 'ErasedVariable' that
  // occurs after 'EraseCall' (the expression that erases the operation). If a
  // use-after-erase is found, returns information about it.
  std::optional<UseAfterErase> find(Stmt *CodeBlock, const Expr *EraseCall,
                                    const DeclRefExpr *ErasedVariable);

private:
  std::optional<UseAfterErase> findInternal(const CFGBlock *Block,
                                            const Expr *EraseCall,
                                            const ValueDecl *ErasedVariable);
  void getUsesAndReinits(const CFGBlock *Block, const ValueDecl *ErasedVariable,
                         SmallVectorImpl<const DeclRefExpr *> *Uses,
                         llvm::SmallPtrSetImpl<const Stmt *> *Reinits);
  void getDeclRefs(const CFGBlock *Block, const Decl *ErasedVariable,
                   llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs);
  void getReinits(const CFGBlock *Block, const Decl *ErasedVariable,
                  llvm::SmallPtrSetImpl<const Stmt *> *Stmts,
                  llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs);

  ASTContext *Context;
  std::unique_ptr<ExprSequence> Sequence;
  std::unique_ptr<StmtToBlockMap> BlockMap;
  llvm::SmallPtrSet<const CFGBlock *, 8> Visited;
};

} // namespace

UseAfterEraseFinder::UseAfterEraseFinder(ASTContext *TheContext)
    : Context(TheContext) {}

std::optional<UseAfterErase>
UseAfterEraseFinder::find(Stmt *CodeBlock, const Expr *EraseCall,
                          const DeclRefExpr *ErasedVariable) {
  // We include implicit and temporary destructors in the CFG so that
  // destructors marked [[noreturn]] are handled correctly in the control flow
  // analysis. (These are used in some styles of assertion macros.)
  CFG::BuildOptions Options;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;
  const auto TheCFG = CFG::buildCFG(nullptr, CodeBlock, Context, Options);
  if (!TheCFG)
    return std::nullopt;

  Sequence = std::make_unique<ExprSequence>(TheCFG.get(), CodeBlock, Context);
  BlockMap = std::make_unique<StmtToBlockMap>(TheCFG.get(), Context);
  Visited.clear();

  const CFGBlock *EraseBlock = BlockMap->blockContainingStmt(EraseCall);
  if (!EraseBlock)
    EraseBlock = &TheCFG->getEntry();

  auto TheUseAfterErase =
      findInternal(EraseBlock, EraseCall, ErasedVariable->getDecl());

  if (TheUseAfterErase) {
    if (const CFGBlock *UseBlock =
            BlockMap->blockContainingStmt(TheUseAfterErase->DeclRef)) {
      // Does the use happen in a later loop iteration than the erase?
      // - If they are in the same CFG block, we know the use happened in a
      //   later iteration if we visited that block a second time.
      // - Otherwise, we know the use happened in a later iteration if the
      //   erase is reachable from the use.
      CFGReverseBlockReachabilityAnalysis CFA(*TheCFG);
      TheUseAfterErase->UseHappensInLaterLoopIteration =
          UseBlock == EraseBlock ? Visited.contains(UseBlock)
                                 : CFA.isReachable(UseBlock, EraseBlock);
    }
  }
  return TheUseAfterErase;
}

std::optional<UseAfterErase>
UseAfterEraseFinder::findInternal(const CFGBlock *Block, const Expr *EraseCall,
                                  const ValueDecl *ErasedVariable) {
  if (Visited.contains(Block))
    return std::nullopt;

  // Mark the block as visited (except if this is the block containing the erase
  // call and it's being visited the first time).
  if (!EraseCall)
    Visited.insert(Block);

  // Get all uses and reinits in the block.
  SmallVector<const DeclRefExpr *, 1> Uses;
  llvm::SmallPtrSet<const Stmt *, 1> Reinits;
  getUsesAndReinits(Block, ErasedVariable, &Uses, &Reinits);

  // Ignore all reinitializations where the erase potentially comes after the
  // reinit.
  SmallVector<const Stmt *, 1> ReinitsToDelete;
  for (const auto *Reinit : Reinits)
    if (EraseCall && Sequence->potentiallyAfter(EraseCall, Reinit))
      ReinitsToDelete.push_back(Reinit);
  for (const auto *Reinit : ReinitsToDelete)
    Reinits.erase(Reinit);

  // Find all uses that potentially come after the erase.
  for (const auto *Use : Uses) {
    if (!EraseCall || Sequence->potentiallyAfter(Use, EraseCall)) {
      // Does the use have a saving reinit? A reinit is saving if it definitely
      // comes before the use, i.e. if there's no potential that the reinit is
      // after the use.
      bool HaveSavingReinit = false;
      for (const auto *Reinit : Reinits)
        if (!Sequence->potentiallyAfter(Reinit, Use))
          HaveSavingReinit = true;

      if (!HaveSavingReinit) {
        UseAfterErase TheUseAfterErase;
        TheUseAfterErase.DeclRef = Use;

        // Is this a use-after-erase that depends on order of evaluation?
        // This is the case if the erase potentially comes after the use (and we
        // already know that the use potentially comes after the erase, which
        // taken together tells us that the ordering is unclear).
        TheUseAfterErase.EvaluationOrderUndefined =
            EraseCall != nullptr && Sequence->potentiallyAfter(EraseCall, Use);

        return TheUseAfterErase;
      }
    }
  }

  // If the operation wasn't reinitialized, call ourselves recursively on all
  // successors.
  if (Reinits.empty()) {
    for (const auto &Succ : Block->succs())
      if (Succ)
        if (auto Found = findInternal(Succ, nullptr, ErasedVariable))
          return Found;
  }

  return std::nullopt;
}

void UseAfterEraseFinder::getUsesAndReinits(
    const CFGBlock *Block, const ValueDecl *ErasedVariable,
    SmallVectorImpl<const DeclRefExpr *> *Uses,
    llvm::SmallPtrSetImpl<const Stmt *> *Reinits) {
  llvm::SmallPtrSet<const DeclRefExpr *, 1> DeclRefs;
  llvm::SmallPtrSet<const DeclRefExpr *, 1> ReinitDeclRefs;

  getDeclRefs(Block, ErasedVariable, &DeclRefs);
  getReinits(Block, ErasedVariable, Reinits, &ReinitDeclRefs);

  // All references to the variable that aren't reinitializations are uses.
  Uses->clear();
  for (const DeclRefExpr *DeclRef : DeclRefs)
    if (!ReinitDeclRefs.contains(DeclRef))
      Uses->push_back(DeclRef);

  // Sort the uses by their occurrence in the source code.
  llvm::sort(*Uses, [](const DeclRefExpr *D1, const DeclRefExpr *D2) {
    return D1->getExprLoc() < D2->getExprLoc();
  });
}

void UseAfterEraseFinder::getDeclRefs(
    const CFGBlock *Block, const Decl *ErasedVariable,
    llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs) {
  DeclRefs->clear();
  const auto DeclRefMatcher =
      declRefExpr(to(equalsNode(ErasedVariable))).bind("declref");

  for (const auto &Elem : *Block) {
    const auto S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    for (const auto &Match :
         match(findAll(DeclRefMatcher), *S->getStmt(), *Context)) {
      const auto *DeclRef = Match.getNodeAs<DeclRefExpr>("declref");
      if (DeclRef && BlockMap->blockContainingStmt(DeclRef) == Block)
        DeclRefs->insert(DeclRef);
    }
  }
}

void UseAfterEraseFinder::getReinits(
    const CFGBlock *Block, const Decl *ErasedVariable,
    llvm::SmallPtrSetImpl<const Stmt *> *Stmts,
    llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs) {
  const auto DeclRefMatcher =
      declRefExpr(to(equalsNode(ErasedVariable))).bind("declref");

  // A reinitialization gives the variable a new, valid operation, which makes
  // subsequent uses safe again. We treat plain assignments and
  // (re-)declarations as reinitializations. For "derived" ops (e.g.
  // `mlir::func::FuncOp`), the assignment is an overloaded `operator=`, which
  // is represented as a `CXXOperatorCallExpr` rather than a `BinaryOperator`.
  const auto ReinitMatcher =
      stmt(anyOf(binaryOperator(hasOperatorName("="),
                                hasLHS(ignoringParenImpCasts(DeclRefMatcher))),
                 cxxOperatorCallExpr(
                     hasOverloadedOperatorName("="),
                     hasArgument(0, ignoringParenImpCasts(DeclRefMatcher))),
                 declStmt(hasDescendant(equalsNode(ErasedVariable)))))
          .bind("reinit");

  Stmts->clear();
  DeclRefs->clear();
  for (const CFGElement &Elem : *Block) {
    const auto S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    for (const auto &Match :
         match(findAll(ReinitMatcher), *S->getStmt(), *Context)) {
      const auto *TheStmt = Match.getNodeAs<Stmt>("reinit");
      const auto *DeclRef = Match.getNodeAs<DeclRefExpr>("declref");
      if (TheStmt && BlockMap->blockContainingStmt(TheStmt) == Block) {
        Stmts->insert(TheStmt);

        // We count DeclStmts as reinitializations, but they don't have a
        // DeclRefExpr associated with them -- so we need to check 'DeclRef'
        // before adding it to the set.
        if (DeclRef)
          DeclRefs->insert(DeclRef);
      }
    }
  }
}

static void emitDiagnostic(const Expr *EraseCall, const DeclRefExpr *EraseArg,
                           const UseAfterErase &Use, ClangTidyCheck *Check) {
  const auto UseLoc = Use.DeclRef->getExprLoc();
  const auto EraseLoc = EraseCall->getExprLoc();

  Check->diag(UseLoc, "operation %0 is used after it was erased")
      << EraseArg->getDecl();
  Check->diag(EraseLoc, "operation erased here", DiagnosticIDs::Note);
  if (Use.EvaluationOrderUndefined) {
    Check->diag(UseLoc,
                "the use and erase are unsequenced, i.e. there is no guarantee "
                "about the order in which they are evaluated",
                DiagnosticIDs::Note);
  } else if (Use.UseHappensInLaterLoopIteration) {
    Check->diag(UseLoc,
                "the use happens in a later loop iteration than the erase",
                DiagnosticIDs::Note);
  }
}

void MlirUseAfterEraseCheck::registerMatchers(MatchFinder *Finder) {
  // The reference to the operation that is being erased.
  const auto Arg = declRefExpr(to(varDecl())).bind("arg");

  // A "derived" op is a value of a subclass of `mlir::OpState` (e.g.
  // `mlir::func::FuncOp`) that wraps an `mlir::Operation *`. The underlying
  // operation can be obtained via the overloaded `operator->` or via
  // `getOperation()`, both of which are declared in `mlir::OpState`.
  const auto OfOpState = ofClass(isSameOrDerivedFrom("::mlir::OpState"));
  const auto UnwrapDerivedOp =
      expr(anyOf(cxxMemberCallExpr(
                     on(ignoringParenImpCasts(Arg)),
                     callee(cxxMethodDecl(hasName("getOperation"), OfOpState))),
                 cxxOperatorCallExpr(hasOverloadedOperatorName("->"),
                                     hasArgument(0, ignoringParenImpCasts(Arg)),
                                     callee(cxxMethodDecl(OfOpState)))));

  // A reference to the operation, either directly (a variable of type
  // `mlir::Operation *`) or through a derived op that is unwrapped to the
  // underlying `mlir::Operation *`.
  const auto OpRef = expr(ignoringParenImpCasts(anyOf(Arg, UnwrapDerivedOp)));

  // Erasure through `mlir::Operation::erase` and `mlir::Operation::destroy`
  // member functions.
  const auto MemberErase = cxxMemberCallExpr(
      on(OpRef), callee(cxxMethodDecl(hasAnyName("erase", "destroy"),
                                      ofClass(hasName("::mlir::Operation")))));

  // Erasure through `mlir::RewriterBase`. The operation that gets invalidated
  // is always the first argument of these methods.
  const auto RewriterErase = cxxMemberCallExpr(
      callee(
          cxxMethodDecl(hasAnyName("eraseOp", "eraseOpResults", "replaceOp",
                                   "replaceOpWithNewOp"),
                        ofClass(isSameOrDerivedFrom("::mlir::RewriterBase")))),
      hasArgument(0, OpRef));

  Finder->addMatcher(
      traverse(TK_AsIs,
               expr(anyOf(MemberErase, RewriterErase),
                    anyOf(hasAncestor(compoundStmt(hasParent(
                              lambdaExpr().bind("containing-lambda")))),
                          hasAncestor(functionDecl().bind("containing-func"))))
                   .bind("erase-call")),
      this);
}

void MlirUseAfterEraseCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *EraseCall = Result.Nodes.getNodeAs<Expr>("erase-call");
  const auto *Arg = Result.Nodes.getNodeAs<DeclRefExpr>("arg");
  const auto *ContainingLambda =
      Result.Nodes.getNodeAs<LambdaExpr>("containing-lambda");
  const auto *ContainingFunc =
      Result.Nodes.getNodeAs<FunctionDecl>("containing-func");

  // Only track operations that are stored in a local variable.
  if (!Arg->getDecl()->getDeclContext()->isFunctionOrMethod())
    return;

  // Collect the code block that could use the operation after it was erased.
  SmallVector<Stmt *> CodeBlocks;
  if (ContainingLambda)
    CodeBlocks.push_back(ContainingLambda->getBody());
  else if (ContainingFunc)
    CodeBlocks.push_back(ContainingFunc->getBody());

  for (Stmt *CodeBlock : CodeBlocks) {
    UseAfterEraseFinder Finder(Result.Context);
    if (auto Use = Finder.find(CodeBlock, EraseCall, Arg))
      emitDiagnostic(EraseCall, Arg, *Use, this);
  }
}

} // namespace clang::tidy::llvm_check
