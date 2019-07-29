//===--- ExtractVariable.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ClangdUnit.h"
#include "Logger.h"
#include "Protocol.h"
#include "Selection.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace {
// information regarding the Expr that is being extracted
class ExtractionContext {
public:
  ExtractionContext(const SelectionTree::Node *Node, const SourceManager &SM,
                    const ASTContext &Ctx);
  const clang::Expr *getExpr() const { return Expr; }
  const SelectionTree::Node *getExprNode() const { return ExprNode; }
  bool isExtractable() const { return Extractable; }
  // The half-open range for the expression to be extracted.
  SourceRange getExtractionChars() const;
  // Generate Replacement for replacing selected expression with given VarName
  tooling::Replacement replaceWithVar(SourceRange Chars,
                                      llvm::StringRef VarName) const;
  // Generate Replacement for declaring the selected Expr as a new variable
  tooling::Replacement insertDeclaration(llvm::StringRef VarName,
                                         SourceRange InitChars) const;

private:
  bool Extractable = false;
  const clang::Expr *Expr;
  const SelectionTree::Node *ExprNode;
  // Stmt before which we will extract
  const clang::Stmt *InsertionPoint = nullptr;
  const SourceManager &SM;
  const ASTContext &Ctx;
  // Decls referenced in the Expr
  std::vector<clang::Decl *> ReferencedDecls;
  // returns true if the Expr doesn't reference any variable declared in scope
  bool exprIsValidOutside(const clang::Stmt *Scope) const;
  // computes the Stmt before which we will extract out Expr
  const clang::Stmt *computeInsertionPoint() const;
};

// Returns all the Decls referenced inside the given Expr
static std::vector<clang::Decl *>
computeReferencedDecls(const clang::Expr *Expr) {
  // RAV subclass to find all DeclRefs in a given Stmt
  class FindDeclRefsVisitor
      : public clang::RecursiveASTVisitor<FindDeclRefsVisitor> {
  public:
    std::vector<Decl *> ReferencedDecls;
    bool VisitDeclRefExpr(DeclRefExpr *DeclRef) { // NOLINT
      ReferencedDecls.push_back(DeclRef->getDecl());
      return true;
    }
  };
  FindDeclRefsVisitor Visitor;
  Visitor.TraverseStmt(const_cast<Stmt *>(dyn_cast<Stmt>(Expr)));
  return Visitor.ReferencedDecls;
}

ExtractionContext::ExtractionContext(const SelectionTree::Node *Node,
                                     const SourceManager &SM,
                                     const ASTContext &Ctx)
    : ExprNode(Node), SM(SM), Ctx(Ctx) {
  Expr = Node->ASTNode.get<clang::Expr>();
  ReferencedDecls = computeReferencedDecls(Expr);
  InsertionPoint = computeInsertionPoint();
  if (InsertionPoint)
    Extractable = true;
}

// checks whether extracting before InsertionPoint will take a
// variable reference out of scope
bool ExtractionContext::exprIsValidOutside(const clang::Stmt *Scope) const {
  SourceLocation ScopeBegin = Scope->getBeginLoc();
  SourceLocation ScopeEnd = Scope->getEndLoc();
  for (const Decl *ReferencedDecl : ReferencedDecls) {
    if (SM.isPointWithin(ReferencedDecl->getBeginLoc(), ScopeBegin, ScopeEnd) &&
        SM.isPointWithin(ReferencedDecl->getEndLoc(), ScopeBegin, ScopeEnd))
      return false;
  }
  return true;
}

// Return the Stmt before which we need to insert the extraction.
// To find the Stmt, we go up the AST Tree and if the Parent of the current
// Stmt is a CompoundStmt, we can extract inside this CompoundStmt just before
// the current Stmt. We ALWAYS insert before a Stmt whose parent is a
// CompoundStmt
//
// FIXME: Extraction from label, switch and case statements
// FIXME: Doens't work for FoldExpr
// FIXME: Ensure extraction from loops doesn't change semantics.
const clang::Stmt *ExtractionContext::computeInsertionPoint() const {
  // returns true if we can extract before InsertionPoint
  auto CanExtractOutside =
      [](const SelectionTree::Node *InsertionPoint) -> bool {
    if (const clang::Stmt *Stmt = InsertionPoint->ASTNode.get<clang::Stmt>()) {
      // Allow all expressions except LambdaExpr since we don't want to extract
      // from the captures/default arguments of a lambda
      if (isa<clang::Expr>(Stmt))
        return !isa<LambdaExpr>(Stmt);
      // We don't yet allow extraction from switch/case stmt as we would need to
      // jump over the switch stmt even if there is a CompoundStmt inside the
      // switch. And there are other Stmts which we don't care about (e.g.
      // continue and break) as there can never be anything to extract from
      // them.
      return isa<AttributedStmt>(Stmt) || isa<CompoundStmt>(Stmt) ||
             isa<CXXForRangeStmt>(Stmt) || isa<DeclStmt>(Stmt) ||
             isa<DoStmt>(Stmt) || isa<ForStmt>(Stmt) || isa<IfStmt>(Stmt) ||
             isa<ReturnStmt>(Stmt) || isa<WhileStmt>(Stmt);
    }
    if (InsertionPoint->ASTNode.get<VarDecl>())
      return true;
    return false;
  };
  for (const SelectionTree::Node *CurNode = getExprNode();
       CurNode->Parent && CanExtractOutside(CurNode);
       CurNode = CurNode->Parent) {
    const clang::Stmt *CurInsertionPoint = CurNode->ASTNode.get<Stmt>();
    // give up if extraction will take a variable out of scope
    if (CurInsertionPoint && !exprIsValidOutside(CurInsertionPoint))
      break;
    if (const clang::Stmt *CurParent = CurNode->Parent->ASTNode.get<Stmt>()) {
      if (isa<CompoundStmt>(CurParent)) {
        // Ensure we don't write inside a macro.
        if (CurParent->getBeginLoc().isMacroID())
          continue;
        return CurInsertionPoint;
      }
    }
  }
  return nullptr;
}

// returns the replacement for substituting the extraction with VarName
tooling::Replacement
ExtractionContext::replaceWithVar(SourceRange Chars,
                                  llvm::StringRef VarName) const {
  unsigned ExtractionLength =
      SM.getFileOffset(Chars.getEnd()) - SM.getFileOffset(Chars.getBegin());
  return tooling::Replacement(SM, Chars.getBegin(), ExtractionLength, VarName);
}
// returns the Replacement for declaring a new variable storing the extraction
tooling::Replacement
ExtractionContext::insertDeclaration(llvm::StringRef VarName,
                                     SourceRange InitializerChars) const {
  llvm::StringRef ExtractionCode = toSourceCode(SM, InitializerChars);
  const SourceLocation InsertionLoc =
      toHalfOpenFileRange(SM, Ctx.getLangOpts(),
                          InsertionPoint->getSourceRange())
          ->getBegin();
  // FIXME: Replace auto with explicit type and add &/&& as necessary
  std::string ExtractedVarDecl = std::string("auto ") + VarName.str() + " = " +
                                 ExtractionCode.str() + "; ";
  return tooling::Replacement(SM, InsertionLoc, 0, ExtractedVarDecl);
}

// Helpers for handling "binary subexpressions" like a + [[b + c]] + d.
//
// These are special, because the formal AST doesn't match what users expect:
// - the AST is ((a + b) + c) + d, so the ancestor expression is `a + b + c`.
// - but extracting `b + c` is reasonable, as + is (mathematically) associative.
//
// So we try to support these cases with some restrictions:
//  - the operator must be associative
//  - no mixing of operators is allowed
//  - we don't look inside macro expansions in the subexpressions
//  - we only adjust the extracted range, so references in the unselected parts
//    of the AST expression (e.g. `a`) are still considered referenced for
//    the purposes of calculating the insertion point.
//    FIXME: it would be nice to exclude these references, by micromanaging
//    the computeReferencedDecls() calls around the binary operator tree.

// Information extracted about a binary operator encounted in a SelectionTree.
// It can represent either an overloaded or built-in operator.
struct ParsedBinaryOperator {
  BinaryOperatorKind Kind;
  SourceLocation ExprLoc;
  llvm::SmallVector<const SelectionTree::Node*, 8> SelectedOperands;

  // If N is a binary operator, populate this and return true.
  bool parse(const SelectionTree::Node &N) {
    SelectedOperands.clear();

    if (const BinaryOperator *Op =
        llvm::dyn_cast_or_null<BinaryOperator>(N.ASTNode.get<Expr>())) {
      Kind = Op->getOpcode();
      ExprLoc = Op->getExprLoc();
      SelectedOperands = N.Children;
      return true;
    }
    if (const CXXOperatorCallExpr *Op =
            llvm::dyn_cast_or_null<CXXOperatorCallExpr>(
                N.ASTNode.get<Expr>())) {
      if (!Op->isInfixBinaryOp())
        return false;

      Kind = BinaryOperator::getOverloadedOpcode(Op->getOperator());
      ExprLoc = Op->getExprLoc();
      // Not all children are args, there's also the callee (operator).
      for (const auto* Child : N.Children) {
        const Expr *E = Child->ASTNode.get<Expr>();
        assert(E && "callee and args should be Exprs!");
        if (E == Op->getArg(0) || E == Op->getArg(1))
          SelectedOperands.push_back(Child);
      }
      return true;
    }
    return false;
  }

  bool associative() const {
    // Must also be left-associative, or update getBinaryOperatorRange()!
    switch (Kind) {
    case BO_Add:
    case BO_Mul:
    case BO_And:
    case BO_Or:
    case BO_Xor:
    case BO_LAnd:
    case BO_LOr:
      return true;
    default:
      return false;
    }
  }

  bool crossesMacroBoundary(const SourceManager &SM) {
    FileID F = SM.getFileID(ExprLoc);
    for (const SelectionTree::Node *Child : SelectedOperands)
      if (SM.getFileID(Child->ASTNode.get<Expr>()->getExprLoc()) != F)
        return true;
    return false;
  }
};

// If have an associative operator at the top level, then we must find
// the start point (rightmost in LHS) and end point (leftmost in RHS).
// We can only descend into subtrees where the operator matches.
//
// e.g. for a + [[b + c]] + d
//        +
//       / \
//  N-> +   d
//     / \
//    +   c <- End
//   / \
//  a   b <- Start
const SourceRange getBinaryOperatorRange(const SelectionTree::Node &N,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts) {
  // If N is not a suitable binary operator, bail out.
  ParsedBinaryOperator Op;
  if (!Op.parse(N.ignoreImplicit()) || !Op.associative() ||
      Op.crossesMacroBoundary(SM) || Op.SelectedOperands.size() != 2)
    return SourceRange();
  BinaryOperatorKind OuterOp = Op.Kind;

  // Because the tree we're interested in contains only one operator type, and
  // all eligible operators are left-associative, the shape of the tree is
  // very restricted: it's a linked list along the left edges.
  // This simplifies our implementation.
  const SelectionTree::Node *Start = Op.SelectedOperands.front(); // LHS
  const SelectionTree::Node *End = Op.SelectedOperands.back();    // RHS
  // End is already correct: it can't be an OuterOp (as it's left-associative).
  // Start needs to be pushed down int the subtree to the right spot.
  while (Op.parse(Start->ignoreImplicit()) && Op.Kind == OuterOp &&
         !Op.crossesMacroBoundary(SM)) {
    assert(!Op.SelectedOperands.empty() && "got only operator on one side!");
    if (Op.SelectedOperands.size() == 1) { // Only Op.RHS selected
      Start = Op.SelectedOperands.back();
      break;
    }
    // Op.LHS is (at least partially) selected, so descend into it.
    Start = Op.SelectedOperands.front();
  }

  return SourceRange(
      toHalfOpenFileRange(SM, LangOpts, Start->ASTNode.getSourceRange())
          ->getBegin(),
      toHalfOpenFileRange(SM, LangOpts, End->ASTNode.getSourceRange())
          ->getEnd());
}

SourceRange ExtractionContext::getExtractionChars() const {
  // Special case: we're extracting an associative binary subexpression.
  SourceRange BinaryOperatorRange =
      getBinaryOperatorRange(*ExprNode, SM, Ctx.getLangOpts());
  if (BinaryOperatorRange.isValid())
    return BinaryOperatorRange;

  // Usual case: we're extracting the whole expression.
  return *toHalfOpenFileRange(SM, Ctx.getLangOpts(), Expr->getSourceRange());
}

/// Extracts an expression to the variable dummy
/// Before:
/// int x = 5 + 4 * 3;
///         ^^^^^
/// After:
/// auto dummy = 5 + 4;
/// int x = dummy * 3;
class ExtractVariable : public Tweak {
public:
  const char *id() const override final;
  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override {
    return "Extract subexpression to variable";
  }
  Intent intent() const override { return Refactor; }
  // Compute the extraction context for the Selection
  bool computeExtractionContext(const SelectionTree::Node *N,
                                const SourceManager &SM, const ASTContext &Ctx);

private:
  // the expression to extract
  std::unique_ptr<ExtractionContext> Target;
};
REGISTER_TWEAK(ExtractVariable)
bool ExtractVariable::prepare(const Selection &Inputs) {
  // we don't trigger on empty selections for now
  if (Inputs.SelectionBegin == Inputs.SelectionEnd)
    return false;
  const ASTContext &Ctx = Inputs.AST.getASTContext();
  const SourceManager &SM = Inputs.AST.getSourceManager();
  const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
  return computeExtractionContext(N, SM, Ctx);
}

Expected<Tweak::Effect> ExtractVariable::apply(const Selection &Inputs) {
  tooling::Replacements Result;
  // FIXME: get variable name from user or suggest based on type
  std::string VarName = "dummy";
  SourceRange Range = Target->getExtractionChars();
  // insert new variable declaration
  if (auto Err = Result.add(Target->insertDeclaration(VarName, Range)))
    return std::move(Err);
  // replace expression with variable name
  if (auto Err = Result.add(Target->replaceWithVar(Range, VarName)))
    return std::move(Err);
  return Effect::applyEdit(Result);
}

// Find the CallExpr whose callee is an ancestor of the DeclRef
const SelectionTree::Node *getCallExpr(const SelectionTree::Node *DeclRef) {
  // we maintain a stack of all exprs encountered while traversing the
  // selectiontree because the callee of the callexpr can be an ancestor of the
  // DeclRef. e.g. Callee can be an ImplicitCastExpr.
  std::vector<const clang::Expr *> ExprStack;
  for (auto *CurNode = DeclRef; CurNode; CurNode = CurNode->Parent) {
    const Expr *CurExpr = CurNode->ASTNode.get<Expr>();
    if (const CallExpr *CallPar = CurNode->ASTNode.get<CallExpr>()) {
      // check whether the callee of the callexpr is present in Expr stack.
      if (std::find(ExprStack.begin(), ExprStack.end(), CallPar->getCallee()) !=
          ExprStack.end())
        return CurNode;
      return nullptr;
    }
    ExprStack.push_back(CurExpr);
  }
  return nullptr;
}

// check if Expr can be assigned to a variable i.e. is non-void type
bool canBeAssigned(const SelectionTree::Node *ExprNode) {
  const clang::Expr *Expr = ExprNode->ASTNode.get<clang::Expr>();
  if (const Type *ExprType = Expr->getType().getTypePtrOrNull())
    // FIXME: check if we need to cover any other types
    return !ExprType->isVoidType();
  return true;
}

// Find the node that will form our ExtractionContext.
// We don't want to trigger for assignment expressions and variable/field
// DeclRefs. For function/member function, we want to extract the entire
// function call.
bool ExtractVariable::computeExtractionContext(const SelectionTree::Node *N,
                                               const SourceManager &SM,
                                               const ASTContext &Ctx) {
  if (!N)
    return false;
  const clang::Expr *SelectedExpr = N->ASTNode.get<clang::Expr>();
  const SelectionTree::Node *TargetNode = N;
  if (!SelectedExpr)
    return false;
  // Extracting Exprs like a = 1 gives dummy = a = 1 which isn't useful.
  if (const BinaryOperator *BinOpExpr =
          dyn_cast_or_null<BinaryOperator>(SelectedExpr)) {
    if (BinOpExpr->isAssignmentOp())
      return false;
  }
  // For function and member function DeclRefs, we look for a parent that is a
  // CallExpr
  if (const DeclRefExpr *DeclRef =
          dyn_cast_or_null<DeclRefExpr>(SelectedExpr)) {
    // Extracting just a variable isn't that useful.
    if (!isa<FunctionDecl>(DeclRef->getDecl()))
      return false;
    TargetNode = getCallExpr(N);
  }
  if (const MemberExpr *Member = dyn_cast_or_null<MemberExpr>(SelectedExpr)) {
    // Extracting just a field member isn't that useful.
    if (!isa<CXXMethodDecl>(Member->getMemberDecl()))
      return false;
    TargetNode = getCallExpr(N);
  }
  if (!TargetNode || !canBeAssigned(TargetNode))
    return false;
  Target = llvm::make_unique<ExtractionContext>(TargetNode, SM, Ctx);
  return Target->isExtractable();
}

} // namespace
} // namespace clangd
} // namespace clang
