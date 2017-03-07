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

#include "../utils/ExprSequence.h"

using namespace clang::ast_matchers;
using namespace clang::tidy::utils;


namespace clang {
namespace tidy {
namespace misc {

namespace {

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

bool isStandardSmartPointer(const ValueDecl *VD) {
  const Type *TheType = VD->getType().getTypePtrOrNull();
  if (!TheType)
    return false;

  const CXXRecordDecl *RecordDecl = TheType->getAsCXXRecordDecl();
  if (!RecordDecl)
    return false;

  const IdentifierInfo *ID = RecordDecl->getIdentifier();
  if (!ID)
    return false;

  StringRef Name = ID->getName();
  if (Name != "unique_ptr" && Name != "shared_ptr" && Name != "weak_ptr")
    return false;

  return RecordDecl->getDeclContext()->isStdNamespace();
}

void UseAfterMoveFinder::getDeclRefs(
    const CFGBlock *Block, const Decl *MovedVariable,
    llvm::SmallPtrSetImpl<const DeclRefExpr *> *DeclRefs) {
  DeclRefs->clear();
  for (const auto &Elem : *Block) {
    Optional<CFGStmt> S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    auto addDeclRefs = [this, Block,
                        DeclRefs](const ArrayRef<BoundNodes> Matches) {
      for (const auto &Match : Matches) {
        const auto *DeclRef = Match.getNodeAs<DeclRefExpr>("declref");
        const auto *Operator = Match.getNodeAs<CXXOperatorCallExpr>("operator");
        if (DeclRef && BlockMap->blockContainingStmt(DeclRef) == Block) {
          // Ignore uses of a standard smart pointer that don't dereference the
          // pointer.
          if (Operator || !isStandardSmartPointer(DeclRef->getDecl())) {
            DeclRefs->insert(DeclRef);
          }
        }
      }
    };

    auto DeclRefMatcher = declRefExpr(hasDeclaration(equalsNode(MovedVariable)),
                                      unless(inDecltypeOrTemplateArg()))
                              .bind("declref");

    addDeclRefs(match(findAll(DeclRefMatcher), *S->getStmt(), *Context));
    addDeclRefs(match(
        findAll(cxxOperatorCallExpr(anyOf(hasOverloadedOperatorName("*"),
                                          hasOverloadedOperatorName("->"),
                                          hasOverloadedOperatorName("[]")),
                                    hasArgument(0, DeclRefMatcher))
                    .bind("operator")),
        *S->getStmt(), *Context));
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

  auto StandardSmartPointerTypeMatcher = hasType(cxxRecordDecl(
      hasAnyName("::std::unique_ptr", "::std::shared_ptr", "::std::weak_ptr")));

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
               // reset() on standard smart pointers.
               cxxMemberCallExpr(
                   on(allOf(DeclRefMatcher, StandardSmartPointerTypeMatcher)),
                   callee(cxxMethodDecl(hasName("reset")))),
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

static void emitDiagnostic(const Expr *MovingCall, const DeclRefExpr *MoveArg,
                           const UseAfterMove &Use, ClangTidyCheck *Check,
                           ASTContext *Context) {
  SourceLocation UseLoc = Use.DeclRef->getExprLoc();
  SourceLocation MoveLoc = MovingCall->getExprLoc();

  Check->diag(UseLoc, "'%0' used after it was moved")
      << MoveArg->getDecl()->getName();
  Check->diag(MoveLoc, "move occurred here", DiagnosticIDs::Note);
  if (Use.EvaluationOrderUndefined) {
    Check->diag(UseLoc,
                "the use and move are unsequenced, i.e. there is no guarantee "
                "about the order in which they are evaluated",
                DiagnosticIDs::Note);
  } else if (UseLoc < MoveLoc || Use.DeclRef == MoveArg) {
    Check->diag(UseLoc,
                "the use happens in a later loop iteration than the move",
                DiagnosticIDs::Note);
  }
}

void UseAfterMoveCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  auto CallMoveMatcher =
      callExpr(callee(functionDecl(hasName("::std::move"))), argumentCountIs(1),
               hasArgument(0, declRefExpr().bind("arg")),
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

  // Ignore the std::move if the variable that was passed to it isn't a local
  // variable.
  if (!Arg->getDecl()->getDeclContext()->isFunctionOrMethod())
    return;

  UseAfterMoveFinder finder(Result.Context);
  UseAfterMove Use;
  if (finder.find(FunctionBody, MovingCall, Arg->getDecl(), &Use))
    emitDiagnostic(MovingCall, Arg, Use, this, Result.Context);
}

} // namespace misc
} // namespace tidy
} // namespace clang
