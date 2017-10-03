//===--- LoopUnrolling.cpp - Unroll loops -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file contains functions which are used to decide if a loop worth to be
/// unrolled. Moreover contains function which mark the CFGBlocks which belongs
/// to the unrolled loop and store them in ProgramState.
///
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CFGStmtMap.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/LoopUnrolling.h"
#include "llvm/ADT/Statistic.h"

using namespace clang;
using namespace ento;
using namespace clang::ast_matchers;

#define DEBUG_TYPE "LoopUnrolling"

STATISTIC(NumTimesLoopUnrolled,
          "The # of times a loop has got completely unrolled");

REGISTER_MAP_WITH_PROGRAMSTATE(UnrolledLoops, const Stmt *,
                               const FunctionDecl *)

namespace clang {
namespace ento {

static bool isLoopStmt(const Stmt *S) {
  return S && (isa<ForStmt>(S) || isa<WhileStmt>(S) || isa<DoStmt>(S));
}

static internal::Matcher<Stmt> simpleCondition(StringRef BindName) {
  return binaryOperator(
      anyOf(hasOperatorName("<"), hasOperatorName(">"), hasOperatorName("<="),
            hasOperatorName(">="), hasOperatorName("!=")),
      hasEitherOperand(ignoringParenImpCasts(
          declRefExpr(to(varDecl(hasType(isInteger())).bind(BindName))))),
      hasEitherOperand(ignoringParenImpCasts(integerLiteral())));
}

static internal::Matcher<Stmt> changeIntBoundNode(StringRef NodeName) {
  return anyOf(hasDescendant(unaryOperator(
                   anyOf(hasOperatorName("--"), hasOperatorName("++")),
                   hasUnaryOperand(ignoringParenImpCasts(
                       declRefExpr(to(varDecl(equalsBoundNode(NodeName)))))))),
               hasDescendant(binaryOperator(
                   anyOf(hasOperatorName("="), hasOperatorName("+="),
                         hasOperatorName("/="), hasOperatorName("*="),
                         hasOperatorName("-=")),
                   hasLHS(ignoringParenImpCasts(
                       declRefExpr(to(varDecl(equalsBoundNode(NodeName)))))))));
}

static internal::Matcher<Stmt> callByRef(StringRef NodeName) {
  return hasDescendant(callExpr(forEachArgumentWithParam(
      declRefExpr(to(varDecl(equalsBoundNode(NodeName)))),
      parmVarDecl(hasType(references(qualType(unless(isConstQualified()))))))));
}

static internal::Matcher<Stmt> assignedToRef(StringRef NodeName) {
  return hasDescendant(varDecl(
      allOf(hasType(referenceType()),
            hasInitializer(
                anyOf(initListExpr(has(
                          declRefExpr(to(varDecl(equalsBoundNode(NodeName)))))),
                      declRefExpr(to(varDecl(equalsBoundNode(NodeName)))))))));
}

static internal::Matcher<Stmt> getAddrTo(StringRef NodeName) {
  return hasDescendant(unaryOperator(
      hasOperatorName("&"),
      hasUnaryOperand(declRefExpr(hasDeclaration(equalsBoundNode(NodeName))))));
}

static internal::Matcher<Stmt> hasSuspiciousStmt(StringRef NodeName) {
  return anyOf(hasDescendant(gotoStmt()), hasDescendant(switchStmt()),
               // Escaping and not known mutation of the loop counter is handled
               // by exclusion of assigning and address-of operators and
               // pass-by-ref function calls on the loop counter from the body.
               changeIntBoundNode(NodeName), callByRef(NodeName),
               getAddrTo(NodeName), assignedToRef(NodeName));
}

static internal::Matcher<Stmt> forLoopMatcher() {
  return forStmt(
             hasCondition(simpleCondition("initVarName")),
             // Initialization should match the form: 'int i = 6' or 'i = 42'.
             hasLoopInit(
                 anyOf(declStmt(hasSingleDecl(
                           varDecl(allOf(hasInitializer(integerLiteral()),
                                         equalsBoundNode("initVarName"))))),
                       binaryOperator(hasLHS(declRefExpr(to(varDecl(
                                          equalsBoundNode("initVarName"))))),
                                      hasRHS(integerLiteral())))),
             // Incrementation should be a simple increment or decrement
             // operator call.
             hasIncrement(unaryOperator(
                 anyOf(hasOperatorName("++"), hasOperatorName("--")),
                 hasUnaryOperand(declRefExpr(
                     to(varDecl(allOf(equalsBoundNode("initVarName"),
                                      hasType(isInteger())))))))),
             unless(hasBody(hasSuspiciousStmt("initVarName")))).bind("forLoop");
}

bool shouldCompletelyUnroll(const Stmt *LoopStmt, ASTContext &ASTCtx) {

  if (!isLoopStmt(LoopStmt))
    return false;

  // TODO: Match the cases where the bound is not a concrete literal but an
  // integer with known value

  auto Matches = match(forLoopMatcher(), *LoopStmt, ASTCtx);
  return !Matches.empty();
}

namespace {
class LoopBlockVisitor : public ConstStmtVisitor<LoopBlockVisitor> {
public:
  LoopBlockVisitor(llvm::SmallPtrSet<const CFGBlock *, 8> &BS) : BlockSet(BS) {}

  void VisitChildren(const Stmt *S) {
    for (const Stmt *Child : S->children())
      if (Child)
        Visit(Child);
  }

  void VisitStmt(const Stmt *S) {
    // In case of nested loops we only unroll the inner loop if it's marked too.
    if (!S || (isLoopStmt(S) && S != LoopStmt))
      return;
    BlockSet.insert(StmtToBlockMap->getBlock(S));
    VisitChildren(S);
  }

  void setBlocksOfLoop(const Stmt *Loop, const CFGStmtMap *M) {
    BlockSet.clear();
    StmtToBlockMap = M;
    LoopStmt = Loop;
    Visit(LoopStmt);
  }

private:
  llvm::SmallPtrSet<const CFGBlock *, 8> &BlockSet;
  const CFGStmtMap *StmtToBlockMap;
  const Stmt *LoopStmt;
};
}
// TODO: refactor this function using LoopExit CFG element - once we have the
// information when the simulation reaches the end of the loop we can cleanup
// the state
bool isUnrolledLoopBlock(const CFGBlock *Block, ExplodedNode *Pred,
                         AnalysisManager &AMgr) {
  const Stmt *Term = Block->getTerminator();
  auto State = Pred->getState();
  // In case of nested loops in an inlined function should not be unrolled only
  // if the inner loop is marked.
  if (Term && isLoopStmt(Term) && !State->contains<UnrolledLoops>(Term))
    return false;

  const CFGBlock *SearchedBlock;
  llvm::SmallPtrSet<const CFGBlock *, 8> BlockSet;
  LoopBlockVisitor LBV(BlockSet);
  // Check the CFGBlocks of every marked loop.
  for (auto &E : State->get<UnrolledLoops>()) {
    SearchedBlock = Block;
    const StackFrameContext *StackFrame = Pred->getStackFrame();
    ParentMap PM(E.second->getBody());
    CFGStmtMap *M = CFGStmtMap::Build(AMgr.getCFG(E.second), &PM);
    LBV.setBlocksOfLoop(E.first, M);
    // In case of an inlined function call check if any of its callSiteBlock is
    // marked.
    while (BlockSet.find(SearchedBlock) == BlockSet.end() && StackFrame) {
      SearchedBlock = StackFrame->getCallSiteBlock();
      if(!SearchedBlock || StackFrame->inTopFrame())
        break;
      StackFrame = StackFrame->getParent()->getCurrentStackFrame();
    }
    delete M;
    if (SearchedBlock)
      return true;
  }
  return false;
}

ProgramStateRef markLoopAsUnrolled(const Stmt *Term, ProgramStateRef State,
                                   const FunctionDecl *FD) {
  if (State->contains<UnrolledLoops>(Term))
    return State;

  State = State->set<UnrolledLoops>(Term, FD);
  ++NumTimesLoopUnrolled;
  return State;
}
}
}
