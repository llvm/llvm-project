//===--- NonConstParameterCheck.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NonConstParameterCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void NonConstParameterCheck::registerMatchers(MatchFinder *Finder) {
  // Add parameters to Parameters.
  Finder->addMatcher(parmVarDecl().bind("Parm"), this);

  // C++ constructor.
  Finder->addMatcher(cxxConstructorDecl().bind("Ctor"), this);

  // Track unused parameters, there is Wunused-parameter about unused
  // parameters.
  Finder->addMatcher(declRefExpr().bind("Ref"), this);

  // Analyse parameter usage in function.
  Finder->addMatcher(stmt(anyOf(unaryOperator(hasAnyOperatorName("++", "--")),
                                binaryOperator(), callExpr(), returnStmt(),
                                cxxConstructExpr()))
                         .bind("Mark"),
                     this);
  Finder->addMatcher(varDecl(hasInitializer(anything())).bind("Mark"), this);
}

void NonConstParameterCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Parm = Result.Nodes.getNodeAs<ParmVarDecl>("Parm")) {
    if (const DeclContext *D = Parm->getParentFunctionOrMethod()) {
      if (const auto *M = dyn_cast<CXXMethodDecl>(D)) {
        if (M->isVirtual() || M->size_overridden_methods() != 0)
          return;
      }
    }
    addParm(Parm);
  } else if (const auto *Ctor =
                 Result.Nodes.getNodeAs<CXXConstructorDecl>("Ctor")) {
    for (const auto *Parm : Ctor->parameters())
      addParm(Parm);
    for (const auto *Init : Ctor->inits())
      markCanNotBeConst(Init->getInit(), true);
  } else if (const auto *Ref = Result.Nodes.getNodeAs<DeclRefExpr>("Ref")) {
    setReferenced(Ref);
  } else if (const auto *S = Result.Nodes.getNodeAs<Stmt>("Mark")) {
    if (const auto *B = dyn_cast<BinaryOperator>(S)) {
      if (B->isAssignmentOp())
        markCanNotBeConst(B, false);
    } else if (const auto *CE = dyn_cast<CallExpr>(S)) {
      // Typically, if a parameter is const then it is fine to make the data
      // const. But sometimes the data is written even though the parameter
      // is const. Mark all data passed by address to the function.
      for (const auto *Arg : CE->arguments()) {
        markCanNotBeConst(Arg->IgnoreParenCasts(), true);
      }

      // Data passed by nonconst reference should not be made const.
      if (const FunctionDecl *FD = CE->getDirectCallee()) {
        unsigned ArgNr = 0U;
        for (const auto *Par : FD->parameters()) {
          if (ArgNr >= CE->getNumArgs())
            break;
          const Expr *Arg = CE->getArg(ArgNr++);
          // Is this a non constant reference parameter?
          const Type *ParType = Par->getType().getTypePtr();
          if (!ParType->isReferenceType() || Par->getType().isConstQualified())
            continue;
          markCanNotBeConst(Arg->IgnoreParenCasts(), false);
        }
      }
    } else if (const auto *CE = dyn_cast<CXXConstructExpr>(S)) {
      for (const auto *Arg : CE->arguments()) {
        markCanNotBeConst(Arg->IgnoreParenCasts(), true);
      }
    } else if (const auto *R = dyn_cast<ReturnStmt>(S)) {
      markCanNotBeConst(R->getRetValue(), true);
    } else if (const auto *U = dyn_cast<UnaryOperator>(S)) {
      markCanNotBeConst(U, true);
    }
  } else if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("Mark")) {
    const QualType T = VD->getType();
    if ((T->isPointerType() && !T->getPointeeType().isConstQualified()) ||
        T->isArrayType())
      markCanNotBeConst(VD->getInit(), true);
  }
}

void NonConstParameterCheck::addParm(const ParmVarDecl *Parm) {
  // Only add nonconst integer/float pointer parameters.
  const QualType T = Parm->getType();
  if (!T->isPointerType() || T->getPointeeType().isConstQualified() ||
      !(T->getPointeeType()->isIntegerType() ||
        T->getPointeeType()->isFloatingType()))
    return;

  if (Parameters.find(Parm) != Parameters.end())
    return;

  ParmInfo PI;
  PI.IsReferenced = false;
  PI.CanBeConst = true;
  Parameters[Parm] = PI;
}

void NonConstParameterCheck::setReferenced(const DeclRefExpr *Ref) {
  auto It = Parameters.find(dyn_cast<ParmVarDecl>(Ref->getDecl()));
  if (It != Parameters.end())
    It->second.IsReferenced = true;
}

void NonConstParameterCheck::onEndOfTranslationUnit() {
  diagnoseNonConstParameters();
}

void NonConstParameterCheck::diagnoseNonConstParameters() {
  for (const auto &It : Parameters) {
    const ParmVarDecl *Par = It.first;
    const ParmInfo &ParamInfo = It.second;

    // Unused parameter => there are other warnings about this.
    if (!ParamInfo.IsReferenced)
      continue;

    // Parameter can't be const.
    if (!ParamInfo.CanBeConst)
      continue;

    SmallVector<FixItHint, 8> Fixes;
    auto *Function =
        dyn_cast_or_null<const FunctionDecl>(Par->getParentFunctionOrMethod());
    if (!Function)
      continue;
    unsigned Index = Par->getFunctionScopeIndex();
    for (FunctionDecl *FnDecl : Function->redecls())
      Fixes.push_back(FixItHint::CreateInsertion(
          FnDecl->getParamDecl(Index)->getBeginLoc(), "const "));

    diag(Par->getLocation(), "pointer parameter '%0' can be pointer to const")
        << Par->getName() << Fixes;
  }
}

void NonConstParameterCheck::markCanNotBeConst(const Expr *E,
                                               bool CanNotBeConst) {
  if (!E)
    return;

  if (const auto *Cast = dyn_cast<ImplicitCastExpr>(E)) {
    // If expression is const then ignore usage.
    const QualType T = Cast->getType();
    if (T->isPointerType() && T->getPointeeType().isConstQualified())
      return;
  }

  E = E->IgnoreParenCasts();

  if (const auto *B = dyn_cast<BinaryOperator>(E)) {
    if (B->isAdditiveOp()) {
      // p + 2
      markCanNotBeConst(B->getLHS(), CanNotBeConst);
      markCanNotBeConst(B->getRHS(), CanNotBeConst);
    } else if (B->isAssignmentOp()) {
      markCanNotBeConst(B->getLHS(), false);

      // If LHS is not const then RHS can't be const.
      const QualType T = B->getLHS()->getType();
      if (T->isPointerType() && !T->getPointeeType().isConstQualified())
        markCanNotBeConst(B->getRHS(), true);
    }
  } else if (const auto *C = dyn_cast<ConditionalOperator>(E)) {
    markCanNotBeConst(C->getTrueExpr(), CanNotBeConst);
    markCanNotBeConst(C->getFalseExpr(), CanNotBeConst);
  } else if (const auto *U = dyn_cast<UnaryOperator>(E)) {
    if (U->getOpcode() == UO_PreInc || U->getOpcode() == UO_PreDec ||
        U->getOpcode() == UO_PostInc || U->getOpcode() == UO_PostDec) {
      if (const auto *SubU =
              dyn_cast<UnaryOperator>(U->getSubExpr()->IgnoreParenCasts()))
        markCanNotBeConst(SubU->getSubExpr(), true);
      markCanNotBeConst(U->getSubExpr(), CanNotBeConst);
    } else if (U->getOpcode() == UO_Deref) {
      if (!CanNotBeConst)
        markCanNotBeConst(U->getSubExpr(), true);
    } else {
      markCanNotBeConst(U->getSubExpr(), CanNotBeConst);
    }
  } else if (const auto *A = dyn_cast<ArraySubscriptExpr>(E)) {
    markCanNotBeConst(A->getBase(), true);
  } else if (const auto *CLE = dyn_cast<CompoundLiteralExpr>(E)) {
    markCanNotBeConst(CLE->getInitializer(), true);
  } else if (const auto *Constr = dyn_cast<CXXConstructExpr>(E)) {
    for (const auto *Arg : Constr->arguments()) {
      if (const auto *M = dyn_cast<MaterializeTemporaryExpr>(Arg))
        markCanNotBeConst(cast<Expr>(M->getSubExpr()), CanNotBeConst);
    }
  } else if (const auto *ILE = dyn_cast<InitListExpr>(E)) {
    for (unsigned I = 0U; I < ILE->getNumInits(); ++I)
      markCanNotBeConst(ILE->getInit(I), true);
  } else if (CanNotBeConst) {
    // Referencing parameter.
    if (const auto *D = dyn_cast<DeclRefExpr>(E)) {
      auto It = Parameters.find(dyn_cast<ParmVarDecl>(D->getDecl()));
      if (It != Parameters.end())
        It->second.CanBeConst = false;
    }
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
