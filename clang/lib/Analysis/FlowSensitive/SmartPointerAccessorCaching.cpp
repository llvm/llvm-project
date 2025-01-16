#include "clang/Analysis/FlowSensitive/SmartPointerAccessorCaching.h"

#include "clang/AST/CanonicalType.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/OperatorKinds.h"

namespace clang::dataflow {

namespace {

using ast_matchers::callee;
using ast_matchers::cxxMemberCallExpr;
using ast_matchers::cxxMethodDecl;
using ast_matchers::cxxOperatorCallExpr;
using ast_matchers::hasName;
using ast_matchers::hasOverloadedOperatorName;
using ast_matchers::ofClass;
using ast_matchers::parameterCountIs;
using ast_matchers::pointerType;
using ast_matchers::referenceType;
using ast_matchers::returns;

bool hasSmartPointerClassShape(const CXXRecordDecl &RD, bool &HasGet,
                               bool &HasValue) {
  // We may want to cache this search, but in current profiles it hasn't shown
  // up as a hot spot (possibly because there aren't many hits, relatively).
  bool HasArrow = false;
  bool HasStar = false;
  CanQualType StarReturnType, ArrowReturnType, GetReturnType, ValueReturnType;
  for (const auto *MD : RD.methods()) {
    // We only consider methods that are const and have zero parameters.
    // It may be that there is a non-const overload for the method, but
    // there should at least be a const overload as well.
    if (!MD->isConst() || MD->getNumParams() != 0)
      continue;
    switch (MD->getOverloadedOperator()) {
    case OO_Star:
      if (MD->getReturnType()->isReferenceType()) {
        HasStar = true;
        StarReturnType = MD->getReturnType()
                             .getNonReferenceType()
                             ->getCanonicalTypeUnqualified();
      }
      break;
    case OO_Arrow:
      if (MD->getReturnType()->isPointerType()) {
        HasArrow = true;
        ArrowReturnType = MD->getReturnType()
                              ->getPointeeType()
                              ->getCanonicalTypeUnqualified();
      }
      break;
    case OO_None: {
      IdentifierInfo *II = MD->getIdentifier();
      if (II == nullptr)
        continue;
      if (II->isStr("get")) {
        if (MD->getReturnType()->isPointerType()) {
          HasGet = true;
          GetReturnType = MD->getReturnType()
                              ->getPointeeType()
                              ->getCanonicalTypeUnqualified();
        }
      } else if (II->isStr("value")) {
        if (MD->getReturnType()->isReferenceType()) {
          HasValue = true;
          ValueReturnType = MD->getReturnType()
                                .getNonReferenceType()
                                ->getCanonicalTypeUnqualified();
        }
      }
    } break;
    default:
      break;
    }
  }

  if (!HasStar || !HasArrow || StarReturnType != ArrowReturnType)
    return false;
  HasGet = HasGet && (GetReturnType == StarReturnType);
  HasValue = HasValue && (ValueReturnType == StarReturnType);
  return true;
}

} // namespace
} // namespace clang::dataflow

// AST_MATCHER macros create an "internal" namespace, so we put it in
// its own anonymous namespace instead of in clang::dataflow.
namespace {

AST_MATCHER(clang::CXXRecordDecl, smartPointerClassWithGet) {
  bool HasGet = false;
  bool HasValue = false;
  bool HasStarAndArrow =
      clang::dataflow::hasSmartPointerClassShape(Node, HasGet, HasValue);
  return HasStarAndArrow && HasGet;
}

AST_MATCHER(clang::CXXRecordDecl, smartPointerClassWithValue) {
  bool HasGet = false;
  bool HasValue = false;
  bool HasStarAndArrow =
      clang::dataflow::hasSmartPointerClassShape(Node, HasGet, HasValue);
  return HasStarAndArrow && HasValue;
}

AST_MATCHER(clang::CXXRecordDecl, smartPointerClassWithGetOrValue) {
  bool HasGet = false;
  bool HasValue = false;
  bool HasStarAndArrow =
      clang::dataflow::hasSmartPointerClassShape(Node, HasGet, HasValue);
  return HasStarAndArrow && (HasGet || HasValue);
}

} // namespace

namespace clang::dataflow {

ast_matchers::StatementMatcher isSmartPointerLikeOperatorStar() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("*"),
      callee(cxxMethodDecl(parameterCountIs(0), returns(referenceType()),
                           ofClass(smartPointerClassWithGetOrValue()))));
}

ast_matchers::StatementMatcher isSmartPointerLikeOperatorArrow() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("->"),
      callee(cxxMethodDecl(parameterCountIs(0), returns(pointerType()),
                           ofClass(smartPointerClassWithGetOrValue()))));
}

ast_matchers::StatementMatcher isSmartPointerLikeValueMethodCall() {
  return cxxMemberCallExpr(callee(
      cxxMethodDecl(parameterCountIs(0), returns(referenceType()),
                    hasName("value"), ofClass(smartPointerClassWithValue()))));
}

ast_matchers::StatementMatcher isSmartPointerLikeGetMethodCall() {
  return cxxMemberCallExpr(callee(
      cxxMethodDecl(parameterCountIs(0), returns(pointerType()), hasName("get"),
                    ofClass(smartPointerClassWithGet()))));
}

const FunctionDecl *
getCanonicalSmartPointerLikeOperatorCallee(const CallExpr *CE) {
  const FunctionDecl *CanonicalCallee = nullptr;
  const CXXMethodDecl *Callee =
      cast_or_null<CXXMethodDecl>(CE->getDirectCallee());
  if (Callee == nullptr)
    return nullptr;
  const CXXRecordDecl *RD = Callee->getParent();
  if (RD == nullptr)
    return nullptr;
  for (const auto *MD : RD->methods()) {
    if (MD->getOverloadedOperator() == OO_Star && MD->isConst() &&
        MD->getNumParams() == 0 && MD->getReturnType()->isReferenceType()) {
      CanonicalCallee = MD;
      break;
    }
  }
  return CanonicalCallee;
}

} // namespace clang::dataflow
