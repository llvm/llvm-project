//===--- UseAfterMoveCheck.cpp - clang-tidy -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseAfterMoveCheck.h"

#include "clang/Analysis/CFG.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {

/// Provides information about the evaluation order of (sub-)expressions within
/// a `CFGBlock`.
///
/// While a `CFGBlock` does contain individual `CFGElement`s for some
/// sub-expressions, the order in which those `CFGElement`s appear reflects
/// only one possible order in which the sub-expressions may be evaluated.
/// However, we want to warn if any of the potential evaluation orders can lead
/// to a use-after-move, not just the one contained in the `CFGBlock`.
///
/// This class implements only a simplified version of the C++ sequencing rules
/// that is, however, sufficient for the purposes of this check. The main
/// limitation is that we do not distinguish between value computation and side
/// effect -- see the "Implementation" section for more details.
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
///   are sequenced before the execution fo the function.
///
/// - In addition, some `Stmt`s are known to be sequenced before or after
///   their siblings. For example, the `Stmt`s that make up a `CompoundStmt`are
///   all sequenced relative to each other. The function
///   `getSequenceSuccessor()` implements these sequencing rules.
class ExprSequence {
public:
  /// Initializes this `ExprSequence` with sequence information for the given
  /// `CFG`.
  ExprSequence(const CFG *TheCFG, ASTContext *TheContext);

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

/// Contains information about a use-after-move.
struct UseAfterMove {
  // The DeclRefExpr that constituted the use of the object.
  const DeclRefExpr *DeclRef;

  // Is the order in which the move and the use are evaluated undefined?
  bool EvaluationOrderUndefined;
};

/// Finds uses of a variable after a move (and maintains state required by the
/// various internal helper functions).
class UseAfterMoveFinder {
public:
  UseAfterMoveFinder(ASTContext *TheContext);

  // Within the given function body, finds the first use of 'MovedVariable' that
  // occurs after 'MovingCall' (the expression that performs the move). If a
  // use-after-move is found, writes information about it to 'TheUseAfterMove'.
  // Returns whether a use-after-move was found.
  bool find(Stmt *FunctionBody, const Expr *MovingCall,
            const ValueDecl *MovedVariable, UseAfterMove *TheUseAfterMove);

private:
  bool findInternal(const CFGBlock *Block, const Expr *MovingCall,
                    const ValueDecl *MovedVariable,
                    UseAfterMove *TheUseAfterMove);
  void getUsesAndReinits(const CFGBlock *Block, const ValueDecl *MovedVariable,
                         llvm::SmallVectorImpl<const DeclRefExpr *> *Uses,
                         llvm::SmallPtrSetImpl<const Stmt *> *Reinits);
  void getDeclRefs(const CFGBlock *Block, const Decl *MovedVariable,
                   llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs);
  void getReinits(const CFGBlock *Block, const ValueDecl *MovedVariable,
                  llvm::SmallPtrSetImpl<const Stmt *> *Stmts,
                  llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs);

  ASTContext *Context;
  std::unique_ptr<ExprSequence> Sequence;
  std::unique_ptr<StmtToBlockMap> BlockMap;
  llvm::SmallPtrSet<const CFGBlock *, 8> Visited;
};

} // namespace

// Returns the Stmt nodes that are parents of 'S', skipping any potential
// intermediate non-Stmt nodes.
//
// In almost all cases, this function returns a single parent or no parents at
// all.
//
// The case that a Stmt has multiple parents is rare but does actually occur in
// the parts of the AST that we're interested in. Specifically, InitListExpr
// nodes cause ASTContext::getParent() to return multiple parents for certain
// nodes in their subtree because RecursiveASTVisitor visits both the syntactic
// and semantic forms of InitListExpr, and the parent-child relationships are
// different between the two forms.
static SmallVector<const Stmt *, 1> getParentStmts(const Stmt *S,
                                                   ASTContext *Context) {
  SmallVector<const Stmt *, 1> Result;

  ASTContext::DynTypedNodeList Parents = Context->getParents(*S);

  SmallVector<ast_type_traits::DynTypedNode, 1> NodesToProcess(Parents.begin(),
                                                               Parents.end());

  while (!NodesToProcess.empty()) {
    ast_type_traits::DynTypedNode Node = NodesToProcess.back();
    NodesToProcess.pop_back();

    if (const auto *S = Node.get<Stmt>()) {
      Result.push_back(S);
    } else {
      Parents = Context->getParents(Node);
      NodesToProcess.append(Parents.begin(), Parents.end());
    }
  }

  return Result;
}

bool isDescendantOrEqual(const Stmt *Descendant, const Stmt *Ancestor,
                         ASTContext *Context) {
  if (Descendant == Ancestor)
    return true;
  for (const Stmt *Parent : getParentStmts(Descendant, Context)) {
    if (isDescendantOrEqual(Parent, Ancestor, Context))
      return true;
  }

  return false;
}

ExprSequence::ExprSequence(const CFG *TheCFG, ASTContext *TheContext)
    : Context(TheContext) {
  for (const auto &SyntheticStmt : TheCFG->synthetic_stmts()) {
    SyntheticStmtSourceMap[SyntheticStmt.first] = SyntheticStmt.second;
  }
}

bool ExprSequence::inSequence(const Stmt *Before, const Stmt *After) const {
  Before = resolveSyntheticStmt(Before);
  After = resolveSyntheticStmt(After);

  // If 'After' is in the subtree of the siblings that follow 'Before' in the
  // chain of successors, we know that 'After' is sequenced after 'Before'.
  for (const Stmt *Successor = getSequenceSuccessor(Before); Successor;
       Successor = getSequenceSuccessor(Successor)) {
    if (isDescendantOrEqual(After, Successor, Context))
      return true;
  }

  // If 'After' is a parent of 'Before' or is sequenced after one of these
  // parents, we know that it is sequenced after 'Before'.
  for (const Stmt *Parent : getParentStmts(Before, Context)) {
    if (Parent == After || inSequence(Parent, After))
      return true;
  }

  return false;
}

bool ExprSequence::potentiallyAfter(const Stmt *After,
                                    const Stmt *Before) const {
  return !inSequence(After, Before);
}

const Stmt *ExprSequence::getSequenceSuccessor(const Stmt *S) const {
  for (const Stmt *Parent : getParentStmts(S, Context)) {
    if (const auto *BO = dyn_cast<BinaryOperator>(Parent)) {
      // Comma operator: Right-hand side is sequenced after the left-hand side.
      if (BO->getLHS() == S && BO->getOpcode() == BO_Comma)
        return BO->getRHS();
    } else if (const auto *InitList = dyn_cast<InitListExpr>(Parent)) {
      // Initializer list: Each initializer clause is sequenced after the
      // clauses that precede it.
      for (unsigned I = 1; I < InitList->getNumInits(); ++I) {
        if (InitList->getInit(I - 1) == S)
          return InitList->getInit(I);
      }
    } else if (const auto *Compound = dyn_cast<CompoundStmt>(Parent)) {
      // Compound statement: Each sub-statement is sequenced after the
      // statements that precede it.
      const Stmt *Previous = nullptr;
      for (const auto *Child : Compound->body()) {
        if (Previous == S)
          return Child;
        Previous = Child;
      }
    } else if (const auto *TheDeclStmt = dyn_cast<DeclStmt>(Parent)) {
      // Declaration: Every initializer expression is sequenced after the
      // initializer expressions that precede it.
      const Expr *PreviousInit = nullptr;
      for (const Decl *TheDecl : TheDeclStmt->decls()) {
        if (const auto *TheVarDecl = dyn_cast<VarDecl>(TheDecl)) {
          if (const Expr *Init = TheVarDecl->getInit()) {
            if (PreviousInit == S)
              return Init;
            PreviousInit = Init;
          }
        }
      }
    } else if (const auto *ForRange = dyn_cast<CXXForRangeStmt>(Parent)) {
      // Range-based for: Loop variable declaration is sequenced before the
      // body. (We need this rule because these get placed in the same
      // CFGBlock.)
      if (S == ForRange->getLoopVarStmt())
        return ForRange->getBody();
    } else if (const auto *TheIfStmt = dyn_cast<IfStmt>(Parent)) {
      // If statement: If a variable is declared inside the condition, the
      // expression used to initialize the variable is sequenced before the
      // evaluation of the condition.
      if (S == TheIfStmt->getConditionVariableDeclStmt())
        return TheIfStmt->getCond();
    }
  }

  return nullptr;
}

const Stmt *ExprSequence::resolveSyntheticStmt(const Stmt *S) const {
  if (SyntheticStmtSourceMap.count(S))
    return SyntheticStmtSourceMap.lookup(S);
  else
    return S;
}

StmtToBlockMap::StmtToBlockMap(const CFG *TheCFG, ASTContext *TheContext)
    : Context(TheContext) {
  for (const auto *B : *TheCFG) {
    for (const auto &Elem : *B) {
      if (Optional<CFGStmt> S = Elem.getAs<CFGStmt>())
        Map[S->getStmt()] = B;
    }
  }
}

const CFGBlock *StmtToBlockMap::blockContainingStmt(const Stmt *S) const {
  while (!Map.count(S)) {
    SmallVector<const Stmt *, 1> Parents = getParentStmts(S, Context);
    if (Parents.empty())
      return nullptr;
    S = Parents[0];
  }

  return Map.lookup(S);
}

// Matches nodes that are
// - Part of a decltype argument or class template argument (we check this by
//   seeing if they are children of a TypeLoc), or
// - Part of a function template argument (we check this by seeing if they are
//   children of a DeclRefExpr that references a function template).
// DeclRefExprs that fulfill these conditions should not be counted as a use or
// move.
static StatementMatcher inDecltypeOrTemplateArg() {
  return anyOf(hasAncestor(typeLoc()),
               hasAncestor(declRefExpr(
                   to(functionDecl(ast_matchers::isTemplateInstantiation())))));
}

UseAfterMoveFinder::UseAfterMoveFinder(ASTContext *TheContext)
    : Context(TheContext) {}

bool UseAfterMoveFinder::find(Stmt *FunctionBody, const Expr *MovingCall,
                              const ValueDecl *MovedVariable,
                              UseAfterMove *TheUseAfterMove) {
  // Generate the CFG manually instead of through an AnalysisDeclContext because
  // it seems the latter can't be used to generate a CFG for the body of a
  // labmda.
  //
  // We include implicit and temporary destructors in the CFG so that
  // destructors marked [[noreturn]] are handled correctly in the control flow
  // analysis. (These are used in some styles of assertion macros.)
  CFG::BuildOptions Options;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;
  std::unique_ptr<CFG> TheCFG =
      CFG::buildCFG(nullptr, FunctionBody, Context, Options);
  if (!TheCFG)
    return false;

  Sequence.reset(new ExprSequence(TheCFG.get(), Context));
  BlockMap.reset(new StmtToBlockMap(TheCFG.get(), Context));
  Visited.clear();

  const CFGBlock *Block = BlockMap->blockContainingStmt(MovingCall);
  if (!Block)
    return false;

  return findInternal(Block, MovingCall, MovedVariable, TheUseAfterMove);
}

bool UseAfterMoveFinder::findInternal(const CFGBlock *Block,
                                      const Expr *MovingCall,
                                      const ValueDecl *MovedVariable,
                                      UseAfterMove *TheUseAfterMove) {
  if (Visited.count(Block))
    return false;

  // Mark the block as visited (except if this is the block containing the
  // std::move() and it's being visited the first time).
  if (!MovingCall)
    Visited.insert(Block);

  // Get all uses and reinits in the block.
  llvm::SmallVector<const DeclRefExpr *, 1> Uses;
  llvm::SmallPtrSet<const Stmt *, 1> Reinits;
  getUsesAndReinits(Block, MovedVariable, &Uses, &Reinits);

  // Ignore all reinitializations where the move potentially comes after the
  // reinit.
  llvm::SmallVector<const Stmt *, 1> ReinitsToDelete;
  for (const Stmt *Reinit : Reinits) {
    if (MovingCall && Sequence->potentiallyAfter(MovingCall, Reinit))
      ReinitsToDelete.push_back(Reinit);
  }
  for (const Stmt *Reinit : ReinitsToDelete) {
    Reinits.erase(Reinit);
  }

  // Find all uses that potentially come after the move.
  for (const DeclRefExpr *Use : Uses) {
    if (!MovingCall || Sequence->potentiallyAfter(Use, MovingCall)) {
      // Does the use have a saving reinit? A reinit is saving if it definitely
      // comes before the use, i.e. if there's no potential that the reinit is
      // after the use.
      bool HaveSavingReinit = false;
      for (const Stmt *Reinit : Reinits) {
        if (!Sequence->potentiallyAfter(Reinit, Use))
          HaveSavingReinit = true;
      }

      if (!HaveSavingReinit) {
        TheUseAfterMove->DeclRef = Use;

        // Is this a use-after-move that depends on order of evaluation?
        // This is the case if the move potentially comes after the use (and we
        // already know that use potentially comes after the move, which taken
        // together tells us that the ordering is unclear).
        TheUseAfterMove->EvaluationOrderUndefined =
            MovingCall != nullptr &&
            Sequence->potentiallyAfter(MovingCall, Use);

        return true;
      }
    }
  }

  // If the object wasn't reinitialized, call ourselves recursively on all
  // successors.
  if (Reinits.empty()) {
    for (const auto &Succ : Block->succs()) {
      if (Succ && findInternal(Succ, nullptr, MovedVariable, TheUseAfterMove))
        return true;
    }
  }

  return false;
}

void UseAfterMoveFinder::getUsesAndReinits(
    const CFGBlock *Block, const ValueDecl *MovedVariable,
    llvm::SmallVectorImpl<const DeclRefExpr *> *Uses,
    llvm::SmallPtrSetImpl<const Stmt *> *Reinits) {
  llvm::SmallPtrSet<const DeclRefExpr *, 1> DeclRefs;
  llvm::SmallPtrSet<const DeclRefExpr *, 1> ReinitDeclRefs;

  getDeclRefs(Block, MovedVariable, &DeclRefs);
  getReinits(Block, MovedVariable, Reinits, &ReinitDeclRefs);

  // All references to the variable that aren't reinitializations are uses.
  Uses->clear();
  for (const DeclRefExpr *DeclRef : DeclRefs) {
    if (!ReinitDeclRefs.count(DeclRef))
      Uses->push_back(DeclRef);
  }

  // Sort the uses by their occurrence in the source code.
  std::sort(Uses->begin(), Uses->end(),
            [](const DeclRefExpr *D1, const DeclRefExpr *D2) {
              return D1->getExprLoc() < D2->getExprLoc();
            });
}

void UseAfterMoveFinder::getDeclRefs(
    const CFGBlock *Block, const Decl *MovedVariable,
    llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs) {
  DeclRefs->clear();
  for (const auto &Elem : *Block) {
    Optional<CFGStmt> S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    SmallVector<BoundNodes, 1> Matches =
        match(findAll(declRefExpr(hasDeclaration(equalsNode(MovedVariable)),
                                  unless(inDecltypeOrTemplateArg()))
                          .bind("declref")),
              *S->getStmt(), *Context);

    for (const auto &Match : Matches) {
      const auto *DeclRef = Match.getNodeAs<DeclRefExpr>("declref");
      if (DeclRef && BlockMap->blockContainingStmt(DeclRef) == Block)
        DeclRefs->insert(DeclRef);
    }
  }
}

void UseAfterMoveFinder::getReinits(
    const CFGBlock *Block, const ValueDecl *MovedVariable,
    llvm::SmallPtrSetImpl<const Stmt *> *Stmts,
    llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs) {
  auto DeclRefMatcher =
      declRefExpr(hasDeclaration(equalsNode(MovedVariable))).bind("declref");

  auto StandardContainerTypeMatcher = hasType(cxxRecordDecl(
      hasAnyName("::std::basic_string", "::std::vector", "::std::deque",
                 "::std::forward_list", "::std::list", "::std::set",
                 "::std::map", "::std::multiset", "::std::multimap",
                 "::std::unordered_set", "::std::unordered_map",
                 "::std::unordered_multiset", "::std::unordered_multimap")));

  // Matches different types of reinitialization.
  auto ReinitMatcher =
      stmt(anyOf(
               // Assignment. In addition to the overloaded assignment operator,
               // test for built-in assignment as well, since template functions
               // may be instantiated to use std::move() on built-in types.
               binaryOperator(hasOperatorName("="), hasLHS(DeclRefMatcher)),
               cxxOperatorCallExpr(hasOverloadedOperatorName("="),
                                   hasArgument(0, DeclRefMatcher)),
               // Declaration. We treat this as a type of reinitialization too,
               // so we don't need to treat it separately.
               declStmt(hasDescendant(equalsNode(MovedVariable))),
               // clear() and assign() on standard containers.
               cxxMemberCallExpr(
                   on(allOf(DeclRefMatcher, StandardContainerTypeMatcher)),
                   // To keep the matcher simple, we check for assign() calls
                   // on all standard containers, even though only vector,
                   // deque, forward_list and list have assign(). If assign()
                   // is called on any of the other containers, this will be
                   // flagged by a compile error anyway.
                   callee(cxxMethodDecl(hasAnyName("clear", "assign")))),
               // Passing variable to a function as a non-const pointer.
               callExpr(forEachArgumentWithParam(
                   unaryOperator(hasOperatorName("&"),
                                 hasUnaryOperand(DeclRefMatcher)),
                   unless(parmVarDecl(hasType(pointsTo(isConstQualified())))))),
               // Passing variable to a function as a non-const lvalue reference
               // (unless that function is std::move()).
               callExpr(forEachArgumentWithParam(
                            DeclRefMatcher,
                            unless(parmVarDecl(hasType(
                                references(qualType(isConstQualified())))))),
                        unless(callee(functionDecl(hasName("::std::move")))))))
          .bind("reinit");

  Stmts->clear();
  DeclRefs->clear();
  for (const auto &Elem : *Block) {
    Optional<CFGStmt> S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    SmallVector<BoundNodes, 1> Matches =
        match(findAll(ReinitMatcher), *S->getStmt(), *Context);

    for (const auto &Match : Matches) {
      const auto *TheStmt = Match.getNodeAs<Stmt>("reinit");
      const auto *TheDeclRef = Match.getNodeAs<DeclRefExpr>("declref");
      if (TheStmt && BlockMap->blockContainingStmt(TheStmt) == Block) {
        Stmts->insert(TheStmt);

        // We count DeclStmts as reinitializations, but they don't have a
        // DeclRefExpr associated with them -- so we need to check 'TheDeclRef'
        // before adding it to the set.
        if (TheDeclRef)
          DeclRefs->insert(TheDeclRef);
      }
    }
  }
}

static void emitDiagnostic(const Expr *MovingCall,
                           const ValueDecl *MovedVariable,
                           const UseAfterMove &Use, ClangTidyCheck *Check,
                           ASTContext *Context) {
  Check->diag(Use.DeclRef->getExprLoc(), "'%0' used after it was moved")
      << MovedVariable->getName();
  Check->diag(MovingCall->getExprLoc(), "move occurred here",
              DiagnosticIDs::Note);
  if (Use.EvaluationOrderUndefined) {
    Check->diag(Use.DeclRef->getExprLoc(),
                "the use and move are unsequenced, i.e. there is no guarantee "
                "about the order in which they are evaluated",
                DiagnosticIDs::Note);
  }
}

void UseAfterMoveCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  auto StandardSmartPointerTypeMatcher = hasType(
      cxxRecordDecl(hasAnyName("::std::unique_ptr", "::std::shared_ptr")));

  auto CallMoveMatcher =
      callExpr(
          callee(functionDecl(hasName("::std::move"))), argumentCountIs(1),
          hasArgument(
              0,
              declRefExpr(unless(StandardSmartPointerTypeMatcher)).bind("arg")),
          anyOf(hasAncestor(lambdaExpr().bind("containing-lambda")),
                hasAncestor(functionDecl().bind("containing-func"))),
          unless(inDecltypeOrTemplateArg()))
          .bind("call-move");

  Finder->addMatcher(
      // To find the Stmt that we assume performs the actual move, we look for
      // the direct ancestor of the std::move() that isn't one of the node
      // types ignored by ignoringParenImpCasts().
      stmt(forEach(expr(ignoringParenImpCasts(CallMoveMatcher))),
           unless(expr(ignoringParenImpCasts(equalsBoundNode("call-move")))))
          .bind("moving-call"),
      this);
}

void UseAfterMoveCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ContainingLambda =
      Result.Nodes.getNodeAs<LambdaExpr>("containing-lambda");
  const auto *ContainingFunc =
      Result.Nodes.getNodeAs<FunctionDecl>("containing-func");
  const auto *CallMove = Result.Nodes.getNodeAs<CallExpr>("call-move");
  const auto *MovingCall = Result.Nodes.getNodeAs<Expr>("moving-call");
  const auto *Arg = Result.Nodes.getNodeAs<DeclRefExpr>("arg");

  if (!MovingCall)
    MovingCall = CallMove;

  Stmt *FunctionBody = nullptr;
  if (ContainingLambda)
    FunctionBody = ContainingLambda->getBody();
  else if (ContainingFunc)
    FunctionBody = ContainingFunc->getBody();
  else
    return;

  const ValueDecl *MovedVariable = Arg->getDecl();

  // Ignore the std::move if the variable that was passed to it isn't a local
  // variable.
  if (!Arg->getDecl()->getDeclContext()->isFunctionOrMethod())
    return;

  UseAfterMoveFinder finder(Result.Context);
  UseAfterMove Use;
  if (finder.find(FunctionBody, MovingCall, MovedVariable, &Use))
    emitDiagnostic(MovingCall, MovedVariable, Use, this, Result.Context);
}

} // namespace misc
} // namespace tidy
} // namespace clang
