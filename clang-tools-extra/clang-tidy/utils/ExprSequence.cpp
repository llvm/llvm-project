//===---------- ExprSequence.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExprSequence.h"
#include "clang/AST/ParentMapContext.h"

namespace clang {
namespace tidy {
namespace utils {

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

  TraversalKindScope RAII(*Context, TK_AsIs);
  DynTypedNodeList Parents = Context->getParents(*S);

  SmallVector<DynTypedNode, 1> NodesToProcess(Parents.begin(), Parents.end());

  while (!NodesToProcess.empty()) {
    DynTypedNode Node = NodesToProcess.back();
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

namespace {
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
}

ExprSequence::ExprSequence(const CFG *TheCFG, const Stmt *Root,
                           ASTContext *TheContext)
    : Context(TheContext), Root(Root) {
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
    // If a statement has multiple parents, make sure we're using the parent
    // that lies within the sub-tree under Root.
    if (!isDescendantOrEqual(Parent, Root, Context))
      continue;

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
      // If statement:
      // - Sequence init statement before variable declaration, if present;
      //   before condition evaluation, otherwise.
      // - Sequence variable declaration (along with the expression used to
      //   initialize it) before the evaluation of the condition.
      if (S == TheIfStmt->getInit()) {
        if (TheIfStmt->getConditionVariableDeclStmt() != nullptr)
          return TheIfStmt->getConditionVariableDeclStmt();
        return TheIfStmt->getCond();
      }
      if (S == TheIfStmt->getConditionVariableDeclStmt())
        return TheIfStmt->getCond();
    } else if (const auto *TheSwitchStmt = dyn_cast<SwitchStmt>(Parent)) {
      // Ditto for switch statements.
      if (S == TheSwitchStmt->getInit()) {
        if (TheSwitchStmt->getConditionVariableDeclStmt() != nullptr)
          return TheSwitchStmt->getConditionVariableDeclStmt();
        return TheSwitchStmt->getCond();
      }
      if (S == TheSwitchStmt->getConditionVariableDeclStmt())
        return TheSwitchStmt->getCond();
    } else if (const auto *TheWhileStmt = dyn_cast<WhileStmt>(Parent)) {
      // While statement: Sequence variable declaration (along with the
      // expression used to initialize it) before the evaluation of the
      // condition.
      if (S == TheWhileStmt->getConditionVariableDeclStmt())
        return TheWhileStmt->getCond();
    }
  }

  return nullptr;
}

const Stmt *ExprSequence::resolveSyntheticStmt(const Stmt *S) const {
  if (SyntheticStmtSourceMap.count(S))
    return SyntheticStmtSourceMap.lookup(S);
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

} // namespace utils
} // namespace tidy
} // namespace clang
