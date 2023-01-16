//===--- MakeMemberFunctionConstCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MakeMemberFunctionConstCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

AST_MATCHER(CXXMethodDecl, isStatic) { return Node.isStatic(); }

AST_MATCHER(CXXMethodDecl, hasTrivialBody) { return Node.hasTrivialBody(); }

AST_MATCHER(CXXRecordDecl, hasAnyDependentBases) {
  return Node.hasAnyDependentBases();
}

AST_MATCHER(CXXMethodDecl, isTemplate) {
  return Node.getTemplatedKind() != FunctionDecl::TK_NonTemplate;
}

AST_MATCHER(CXXMethodDecl, isDependentContext) {
  return Node.isDependentContext();
}

AST_MATCHER(CXXMethodDecl, isInsideMacroDefinition) {
  const ASTContext &Ctxt = Finder->getASTContext();
  return clang::Lexer::makeFileCharRange(
             clang::CharSourceRange::getCharRange(
                 Node.getTypeSourceInfo()->getTypeLoc().getSourceRange()),
             Ctxt.getSourceManager(), Ctxt.getLangOpts())
      .isInvalid();
}

AST_MATCHER_P(CXXMethodDecl, hasCanonicalDecl,
              ast_matchers::internal::Matcher<CXXMethodDecl>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getCanonicalDecl(), Finder, Builder);
}

enum UsageKind { Unused, Const, NonConst };

class FindUsageOfThis : public RecursiveASTVisitor<FindUsageOfThis> {
  ASTContext &Ctxt;

public:
  FindUsageOfThis(ASTContext &Ctxt) : Ctxt(Ctxt) {}
  UsageKind Usage = Unused;

  template <class T> const T *getParent(const Expr *E) {
    DynTypedNodeList Parents = Ctxt.getParents(*E);
    if (Parents.size() != 1)
      return nullptr;

    return Parents.begin()->get<T>();
  }

  const Expr *getParentExprIgnoreParens(const Expr *E) {
    const Expr *Parent = getParent<Expr>(E);
    while (isa_and_nonnull<ParenExpr>(Parent))
      Parent = getParent<Expr>(Parent);
    return Parent;
  }

  bool VisitUnresolvedMemberExpr(const UnresolvedMemberExpr *) {
    // An UnresolvedMemberExpr might resolve to a non-const non-static
    // member function.
    Usage = NonConst;
    return false; // Stop traversal.
  }

  bool VisitCXXConstCastExpr(const CXXConstCastExpr *) {
    // Workaround to support the pattern
    // class C {
    //   const S *get() const;
    //   S* get() {
    //     return const_cast<S*>(const_cast<const C*>(this)->get());
    //   }
    // };
    // Here, we don't want to make the second 'get' const even though
    // it only calls a const member function on this.
    Usage = NonConst;
    return false; // Stop traversal.
  }

  // Our AST is
  //  `-ImplicitCastExpr
  //  (possibly `-UnaryOperator Deref)
  //        `-CXXThisExpr 'S *' this
  bool visitUser(const ImplicitCastExpr *Cast) {
    if (Cast->getCastKind() != CK_NoOp)
      return false; // Stop traversal.

    // Only allow NoOp cast to 'const S' or 'const S *'.
    QualType QT = Cast->getType();
    if (QT->isPointerType())
      QT = QT->getPointeeType();

    if (!QT.isConstQualified())
      return false; // Stop traversal.

    const auto *Parent = getParent<Stmt>(Cast);
    if (!Parent)
      return false; // Stop traversal.

    if (isa<ReturnStmt>(Parent))
      return true; // return (const S*)this;

    if (isa<CallExpr>(Parent))
      return true; // use((const S*)this);

    // ((const S*)this)->Member
    if (const auto *Member = dyn_cast<MemberExpr>(Parent))
      return visitUser(Member, /*OnConstObject=*/true);

    return false; // Stop traversal.
  }

  // If OnConstObject is true, then this is a MemberExpr using
  // a constant this, i.e. 'const S' or 'const S *'.
  bool visitUser(const MemberExpr *Member, bool OnConstObject) {
    if (Member->isBoundMemberFunction(Ctxt)) {
      if (!OnConstObject || Member->getFoundDecl().getAccess() != AS_public) {
        // Non-public non-static member functions might not preserve the
        // logical constness. E.g. in
        // class C {
        //   int &data() const;
        // public:
        //   int &get() { return data(); }
        // };
        // get() uses a private const method, but must not be made const
        // itself.
        return false; // Stop traversal.
      }
      // Using a public non-static const member function.
      return true;
    }

    const auto *Parent = getParentExprIgnoreParens(Member);

    if (const auto *Cast = dyn_cast_or_null<ImplicitCastExpr>(Parent)) {
      // A read access to a member is safe when the member either
      // 1) has builtin type (a 'const int' cannot be modified),
      // 2) or it's a public member (the pointee of a public 'int * const' can
      // can be modified by any user of the class).
      if (Member->getFoundDecl().getAccess() != AS_public &&
          !Cast->getType()->isBuiltinType())
        return false;

      if (Cast->getCastKind() == CK_LValueToRValue)
        return true;

      if (Cast->getCastKind() == CK_NoOp && Cast->getType().isConstQualified())
        return true;
    }

    if (const auto *M = dyn_cast_or_null<MemberExpr>(Parent))
      return visitUser(M, /*OnConstObject=*/false);

    return false; // Stop traversal.
  }

  bool VisitCXXThisExpr(const CXXThisExpr *E) {
    Usage = Const;

    const auto *Parent = getParentExprIgnoreParens(E);

    // Look through deref of this.
    if (const auto *UnOp = dyn_cast_or_null<UnaryOperator>(Parent)) {
      if (UnOp->getOpcode() == UO_Deref) {
        Parent = getParentExprIgnoreParens(UnOp);
      }
    }

    // It's okay to
    //  return (const S*)this;
    //  use((const S*)this);
    //  ((const S*)this)->f()
    // when 'f' is a public member function.
    if (const auto *Cast = dyn_cast_or_null<ImplicitCastExpr>(Parent)) {
      if (visitUser(Cast))
        return true;

      // And it's also okay to
      //   (const T)(S->t)
      //   (LValueToRValue)(S->t)
      // when 't' is either of builtin type or a public member.
    } else if (const auto *Member = dyn_cast_or_null<MemberExpr>(Parent)) {
      if (visitUser(Member, /*OnConstObject=*/false))
        return true;
    }

    // Unknown user of this.
    Usage = NonConst;
    return false; // Stop traversal.
  }
};

AST_MATCHER(CXXMethodDecl, usesThisAsConst) {
  FindUsageOfThis UsageOfThis(Finder->getASTContext());

  // TraverseStmt does not modify its argument.
  UsageOfThis.TraverseStmt(const_cast<Stmt *>(Node.getBody()));

  return UsageOfThis.Usage == Const;
}

void MakeMemberFunctionConstCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          cxxMethodDecl(
              isDefinition(), isUserProvided(),
              unless(anyOf(
                  isExpansionInSystemHeader(), isVirtual(), isConst(),
                  isStatic(), hasTrivialBody(), cxxConstructorDecl(),
                  cxxDestructorDecl(), isTemplate(), isDependentContext(),
                  ofClass(anyOf(isLambda(),
                                hasAnyDependentBases()) // Method might become
                                                        // virtual depending on
                                                        // template base class.
                          ),
                  isInsideMacroDefinition(),
                  hasCanonicalDecl(isInsideMacroDefinition()))),
              usesThisAsConst())
              .bind("x")),
      this);
}

static SourceLocation getConstInsertionPoint(const CXXMethodDecl *M) {
  TypeSourceInfo *TSI = M->getTypeSourceInfo();
  if (!TSI)
    return {};

  FunctionTypeLoc FTL =
      TSI->getTypeLoc().IgnoreParens().getAs<FunctionTypeLoc>();
  if (!FTL)
    return {};

  return FTL.getRParenLoc().getLocWithOffset(1);
}

void MakeMemberFunctionConstCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Definition = Result.Nodes.getNodeAs<CXXMethodDecl>("x");

  const auto *Declaration = Definition->getCanonicalDecl();

  auto Diag = diag(Definition->getLocation(), "method %0 can be made const")
              << Definition
              << FixItHint::CreateInsertion(getConstInsertionPoint(Definition),
                                            " const");
  if (Declaration != Definition) {
    Diag << FixItHint::CreateInsertion(getConstInsertionPoint(Declaration),
                                       " const");
  }
}

} // namespace clang::tidy::readability
