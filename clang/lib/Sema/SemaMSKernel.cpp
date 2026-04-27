#include "TreeTransform.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaConsumer.h"

using namespace clang;

namespace {
ExprResult TransformUnaryOperator(Sema &SemaRef, UnaryOperator *UO) {
  if (UO->getOpcode() != UO_AddrOf)
    return {};

  SmallVector<Sema::OffsetOfComponent, 4> Components;
  Expr *Current = UO->getSubExpr()->IgnoreParens();
  while (true) {
    // Strip away the "noise" added by array decays or parentheses
    Current = Current->IgnoreParenImpCasts();
    Sema::OffsetOfComponent Comp;
    Comp.LocStart = Current->getBeginLoc();
    Comp.LocEnd = Current->getEndLoc();

    if (auto *ME = dyn_cast<MemberExpr>(Current)) {
      Comp.isBrackets = false;
      Comp.U.IdentInfo = ME->getMemberDecl()->getIdentifier();
      Components.push_back(Comp);
      Current = ME->getBase();
    } else if (auto *ASE = dyn_cast<ArraySubscriptExpr>(Current)) {
      Comp.isBrackets = true;
      // In offsetof, the index must be an expression
      Comp.U.E = ASE->getIdx();
      Components.push_back(Comp);
      Current = ASE->getBase();
    } else {
      // No more members or subscripts
      break;
    }
  }
  // Verify we ended at a Null Pointer Cast
  Expr *Base = Current->IgnoreParenCasts();
  if (!Components.empty() &&
      Base->isNullPointerConstant(SemaRef.Context,
                                  Expr::NPC_ValueDependentIsNotNull)) {
    // Don't treat &((MyStruct*)0)[1] as an offsetof expression
    if (Components.back().isBrackets)
      return {};
    // Targets like amdgcn, where nullptr != 0, are ignored
    if (SemaRef.Context.getTargetNullPointerValue(Current->getType()))
      return {};
    std::reverse(Components.begin(), Components.end());

    // Get the root structure type
    TypeSourceInfo *TInfo = SemaRef.Context.getTrivialTypeSourceInfo(
        Current->getType()->getPointeeType(), Current->getBeginLoc());
    return SemaRef.BuildBuiltinOffsetOf(UO->getBeginLoc(), TInfo, Components,
                                        UO->getEndLoc());
  }

  return {};
}
} // end anonymous namespace

ExprResult clang::Sema::TransformForMSKernel(Expr *E) {
  auto *UO = dyn_cast_or_null<UnaryOperator>(E);
  if (!UO)
    return {};
  ExprResult NewUO = TransformUnaryOperator(*this, UO);
  return NewUO.isUsable() ? NewUO : ExprResult();
}
