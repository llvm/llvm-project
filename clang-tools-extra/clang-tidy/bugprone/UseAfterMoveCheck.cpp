//===--- UseAfterMoveCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseAfterMoveCheck.h"

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"

#include "../utils/ExprSequence.h"
#include "../utils/Matchers.h"
#include <optional>

using namespace clang::ast_matchers;
using namespace clang::tidy::utils;

namespace clang::tidy::bugprone {

using matchers::hasUnevaluatedContext;

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

  // Within the given code block, finds the first use of 'MovedVariable' that
  // occurs after 'MovingCall' (the expression that performs the move). If a
  // use-after-move is found, writes information about it to 'TheUseAfterMove'.
  // Returns whether a use-after-move was found.
  bool find(Stmt *CodeBlock, const Expr *MovingCall,
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
                   to(functionDecl(ast_matchers::isTemplateInstantiation())))),
               hasAncestor(expr(hasUnevaluatedContext())));
}

UseAfterMoveFinder::UseAfterMoveFinder(ASTContext *TheContext)
    : Context(TheContext) {}

bool UseAfterMoveFinder::find(Stmt *CodeBlock, const Expr *MovingCall,
                              const ValueDecl *MovedVariable,
                              UseAfterMove *TheUseAfterMove) {
  // Generate the CFG manually instead of through an AnalysisDeclContext because
  // it seems the latter can't be used to generate a CFG for the body of a
  // lambda.
  //
  // We include implicit and temporary destructors in the CFG so that
  // destructors marked [[noreturn]] are handled correctly in the control flow
  // analysis. (These are used in some styles of assertion macros.)
  CFG::BuildOptions Options;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;
  std::unique_ptr<CFG> TheCFG =
      CFG::buildCFG(nullptr, CodeBlock, Context, Options);
  if (!TheCFG)
    return false;

  Sequence = std::make_unique<ExprSequence>(TheCFG.get(), CodeBlock, Context);
  BlockMap = std::make_unique<StmtToBlockMap>(TheCFG.get(), Context);
  Visited.clear();

  const CFGBlock *Block = BlockMap->blockContainingStmt(MovingCall);
  if (!Block) {
    // This can happen if MovingCall is in a constructor initializer, which is
    // not included in the CFG because the CFG is built only from the function
    // body.
    Block = &TheCFG->getEntry();
  }

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
  // If `Reinit` is identical to `MovingCall`, we're looking at a move-to-self
  // (e.g. `a = std::move(a)`). Count these as reinitializations.
  llvm::SmallVector<const Stmt *, 1> ReinitsToDelete;
  for (const Stmt *Reinit : Reinits) {
    if (MovingCall && Reinit != MovingCall &&
        Sequence->potentiallyAfter(MovingCall, Reinit))
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
  llvm::sort(*Uses, [](const DeclRefExpr *D1, const DeclRefExpr *D2) {
    return D1->getExprLoc() < D2->getExprLoc();
  });
}

bool isStandardSmartPointer(const ValueDecl *VD) {
  const Type *TheType = VD->getType().getNonReferenceType().getTypePtrOrNull();
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
    std::optional<CFGStmt> S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    auto AddDeclRefs = [this, Block,
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

    AddDeclRefs(match(traverse(TK_AsIs, findAll(DeclRefMatcher)), *S->getStmt(),
                      *Context));
    AddDeclRefs(match(findAll(cxxOperatorCallExpr(
                                  hasAnyOverloadedOperatorName("*", "->", "[]"),
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

  auto StandardContainerTypeMatcher = hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(hasAnyName(
          "::std::basic_string", "::std::vector", "::std::deque",
          "::std::forward_list", "::std::list", "::std::set", "::std::map",
          "::std::multiset", "::std::multimap", "::std::unordered_set",
          "::std::unordered_map", "::std::unordered_multiset",
          "::std::unordered_multimap"))))));

  auto StandardSmartPointerTypeMatcher = hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(hasAnyName(
          "::std::unique_ptr", "::std::shared_ptr", "::std::weak_ptr"))))));

  // Matches different types of reinitialization.
  auto ReinitMatcher =
      stmt(anyOf(
               // Assignment. In addition to the overloaded assignment operator,
               // test for built-in assignment as well, since template functions
               // may be instantiated to use std::move() on built-in types.
               binaryOperation(hasOperatorName("="), hasLHS(DeclRefMatcher)),
               // Declaration. We treat this as a type of reinitialization too,
               // so we don't need to treat it separately.
               declStmt(hasDescendant(equalsNode(MovedVariable))),
               // clear() and assign() on standard containers.
               cxxMemberCallExpr(
                   on(expr(DeclRefMatcher, StandardContainerTypeMatcher)),
                   // To keep the matcher simple, we check for assign() calls
                   // on all standard containers, even though only vector,
                   // deque, forward_list and list have assign(). If assign()
                   // is called on any of the other containers, this will be
                   // flagged by a compile error anyway.
                   callee(cxxMethodDecl(hasAnyName("clear", "assign")))),
               // reset() on standard smart pointers.
               cxxMemberCallExpr(
                   on(expr(DeclRefMatcher, StandardSmartPointerTypeMatcher)),
                   callee(cxxMethodDecl(hasName("reset")))),
               // Methods that have the [[clang::reinitializes]] attribute.
               cxxMemberCallExpr(
                   on(DeclRefMatcher),
                   callee(cxxMethodDecl(hasAttr(clang::attr::Reinitializes)))),
               // Passing variable to a function as a non-const pointer.
               callExpr(forEachArgumentWithParam(
                   unaryOperator(hasOperatorName("&"),
                                 hasUnaryOperand(DeclRefMatcher)),
                   unless(parmVarDecl(hasType(pointsTo(isConstQualified())))))),
               // Passing variable to a function as a non-const lvalue reference
               // (unless that function is std::move()).
               callExpr(forEachArgumentWithParam(
                            traverse(TK_AsIs, DeclRefMatcher),
                            unless(parmVarDecl(hasType(
                                references(qualType(isConstQualified())))))),
                        unless(callee(functionDecl(
                            hasAnyName("::std::move", "::std::forward")))))))
          .bind("reinit");

  Stmts->clear();
  DeclRefs->clear();
  for (const auto &Elem : *Block) {
    std::optional<CFGStmt> S = Elem.getAs<CFGStmt>();
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

enum class MoveType {
  Move,    // std::move
  Forward, // std::forward
};

static MoveType determineMoveType(const FunctionDecl *FuncDecl) {
  if (FuncDecl->getName() == "move")
    return MoveType::Move;
  if (FuncDecl->getName() == "forward")
    return MoveType::Forward;

  llvm_unreachable("Invalid move type");
}

static void emitDiagnostic(const Expr *MovingCall, const DeclRefExpr *MoveArg,
                           const UseAfterMove &Use, ClangTidyCheck *Check,
                           ASTContext *Context, MoveType Type) {
  const SourceLocation UseLoc = Use.DeclRef->getExprLoc();
  const SourceLocation MoveLoc = MovingCall->getExprLoc();

  const bool IsMove = (Type == MoveType::Move);

  Check->diag(UseLoc, "'%0' used after it was %select{forwarded|moved}1")
      << MoveArg->getDecl()->getName() << IsMove;
  Check->diag(MoveLoc, "%select{forward|move}0 occurred here",
              DiagnosticIDs::Note)
      << IsMove;
  if (Use.EvaluationOrderUndefined) {
    Check->diag(
        UseLoc,
        "the use and %select{forward|move}0 are unsequenced, i.e. "
        "there is no guarantee about the order in which they are evaluated",
        DiagnosticIDs::Note)
        << IsMove;
  } else if (UseLoc < MoveLoc || Use.DeclRef == MoveArg) {
    Check->diag(UseLoc,
                "the use happens in a later loop iteration than the "
                "%select{forward|move}0",
                DiagnosticIDs::Note)
        << IsMove;
  }
}

void UseAfterMoveCheck::registerMatchers(MatchFinder *Finder) {
  // try_emplace is a common maybe-moving function that returns a
  // bool to tell callers whether it moved. Ignore std::move inside
  // try_emplace to avoid false positives as we don't track uses of
  // the bool.
  auto TryEmplaceMatcher =
      cxxMemberCallExpr(callee(cxxMethodDecl(hasName("try_emplace"))));
  auto CallMoveMatcher =
      callExpr(argumentCountIs(1),
               callee(functionDecl(hasAnyName("::std::move", "::std::forward"))
                          .bind("move-decl")),
               hasArgument(0, declRefExpr().bind("arg")),
               unless(inDecltypeOrTemplateArg()),
               unless(hasParent(TryEmplaceMatcher)), expr().bind("call-move"),
               anyOf(hasAncestor(compoundStmt(
                         hasParent(lambdaExpr().bind("containing-lambda")))),
                     hasAncestor(functionDecl(anyOf(
                         cxxConstructorDecl(
                             hasAnyConstructorInitializer(withInitializer(
                                 expr(anyOf(equalsBoundNode("call-move"),
                                            hasDescendant(expr(
                                                equalsBoundNode("call-move")))))
                                     .bind("containing-ctor-init"))))
                             .bind("containing-ctor"),
                         functionDecl().bind("containing-func"))))));

  Finder->addMatcher(
      traverse(
          TK_AsIs,
          // To find the Stmt that we assume performs the actual move, we look
          // for the direct ancestor of the std::move() that isn't one of the
          // node types ignored by ignoringParenImpCasts().
          stmt(
              forEach(expr(ignoringParenImpCasts(CallMoveMatcher))),
              // Don't allow an InitListExpr to be the moving call. An
              // InitListExpr has both a syntactic and a semantic form, and the
              // parent-child relationships are different between the two. This
              // could cause an InitListExpr to be analyzed as the moving call
              // in addition to the Expr that we actually want, resulting in two
              // diagnostics with different code locations for the same move.
              unless(initListExpr()),
              unless(expr(ignoringParenImpCasts(equalsBoundNode("call-move")))))
              .bind("moving-call")),
      this);
}

void UseAfterMoveCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ContainingCtor =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("containing-ctor");
  const auto *ContainingCtorInit =
      Result.Nodes.getNodeAs<Expr>("containing-ctor-init");
  const auto *ContainingLambda =
      Result.Nodes.getNodeAs<LambdaExpr>("containing-lambda");
  const auto *ContainingFunc =
      Result.Nodes.getNodeAs<FunctionDecl>("containing-func");
  const auto *CallMove = Result.Nodes.getNodeAs<CallExpr>("call-move");
  const auto *MovingCall = Result.Nodes.getNodeAs<Expr>("moving-call");
  const auto *Arg = Result.Nodes.getNodeAs<DeclRefExpr>("arg");
  const auto *MoveDecl = Result.Nodes.getNodeAs<FunctionDecl>("move-decl");

  if (!MovingCall || !MovingCall->getExprLoc().isValid())
    MovingCall = CallMove;

  // Ignore the std::move if the variable that was passed to it isn't a local
  // variable.
  if (!Arg->getDecl()->getDeclContext()->isFunctionOrMethod())
    return;

  // Collect all code blocks that could use the arg after move.
  llvm::SmallVector<Stmt *> CodeBlocks{};
  if (ContainingCtor) {
    CodeBlocks.push_back(ContainingCtor->getBody());
    if (ContainingCtorInit) {
      // Collect the constructor initializer expressions.
      bool BeforeMove{true};
      for (CXXCtorInitializer *Init : ContainingCtor->inits()) {
        if (BeforeMove && Init->getInit()->IgnoreImplicit() ==
                              ContainingCtorInit->IgnoreImplicit())
          BeforeMove = false;
        if (!BeforeMove)
          CodeBlocks.push_back(Init->getInit());
      }
    }
  } else if (ContainingLambda) {
    CodeBlocks.push_back(ContainingLambda->getBody());
  } else if (ContainingFunc) {
    CodeBlocks.push_back(ContainingFunc->getBody());
  }

  for (Stmt *CodeBlock : CodeBlocks) {
    UseAfterMoveFinder Finder(Result.Context);
    UseAfterMove Use;
    if (Finder.find(CodeBlock, MovingCall, Arg->getDecl(), &Use))
      emitDiagnostic(MovingCall, Arg, Use, this, Result.Context,
                     determineMoveType(MoveDecl));
  }
}

} // namespace clang::tidy::bugprone
