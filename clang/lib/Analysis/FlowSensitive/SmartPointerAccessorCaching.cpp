#include "clang/Analysis/FlowSensitive/SmartPointerAccessorCaching.h"

#include "clang/AST/CanonicalType.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/OperatorKinds.h"

namespace clang::dataflow {

namespace {

using ast_matchers::callee;
using ast_matchers::cxxMemberCallExpr;
using ast_matchers::cxxMethodDecl;
using ast_matchers::cxxOperatorCallExpr;
using ast_matchers::hasCanonicalType;
using ast_matchers::hasName;
using ast_matchers::hasOverloadedOperatorName;
using ast_matchers::ofClass;
using ast_matchers::parameterCountIs;
using ast_matchers::pointerType;
using ast_matchers::referenceType;
using ast_matchers::returns;

CanQualType getLikeReturnType(QualType RT) {
  if (!RT.isNull() && RT->isPointerType()) {
    return RT->getPointeeType()
        ->getCanonicalTypeUnqualified()
        .getUnqualifiedType();
  }
  return {};
}

CanQualType valueLikeReturnType(QualType RT) {
  if (!RT.isNull() && RT->isReferenceType()) {
    return RT.getNonReferenceType()
        ->getCanonicalTypeUnqualified()
        .getUnqualifiedType();
  }
  return {};
}

CanQualType pointerLikeReturnType(const CXXRecordDecl &RD) {
  // We may want to cache this search, but in current profiles it hasn't shown
  // up as a hot spot (possibly because there aren't many hits, relatively).
  CanQualType StarReturnType, ArrowReturnType;
  for (const auto *MD : RD.methods()) {
    // We only consider methods that are const and have zero parameters.
    // It may be that there is a non-const overload for the method, but
    // there should at least be a const overload as well.
    if (!MD->isConst() || MD->getNumParams() != 0)
      continue;
    switch (MD->getOverloadedOperator()) {
    case OO_Star:
      StarReturnType = valueLikeReturnType(MD->getReturnType());
      break;
    case OO_Arrow:
      ArrowReturnType = getLikeReturnType(MD->getReturnType());
      break;
    default:
      break;
    }
  }
  if (!StarReturnType.isNull() && !ArrowReturnType.isNull() &&
      StarReturnType == ArrowReturnType)
    return StarReturnType;

  return {};
}

QualType findReturnType(const CXXRecordDecl &RD, StringRef MethodName) {
  for (const auto *MD : RD.methods()) {
    // We only consider methods that are const and have zero parameters.
    // It may be that there is a non-const overload for the method, but
    // there should at least be a const overload as well.
    if (!MD->isConst() || MD->getNumParams() != 0 ||
        MD->getOverloadedOperator() != OO_None)
      continue;
    clang::IdentifierInfo *II = MD->getIdentifier();
    if (II && II->isStr(MethodName))
      return MD->getReturnType();
  }
  return {};
}

} // namespace
} // namespace clang::dataflow

// AST_MATCHER macros create an "internal" namespace, so we put it in
// its own anonymous namespace instead of in clang::dataflow.
namespace {

using clang::dataflow::findReturnType;
using clang::dataflow::getLikeReturnType;
using clang::dataflow::pointerLikeReturnType;
using clang::dataflow::valueLikeReturnType;

AST_MATCHER_P(clang::CXXRecordDecl, smartPointerClassWithGetLike,
              clang::StringRef, MethodName) {
  auto RT = pointerLikeReturnType(Node);
  if (RT.isNull())
    return false;
  return getLikeReturnType(findReturnType(Node, MethodName)) == RT;
}

AST_MATCHER_P(clang::CXXRecordDecl, smartPointerClassWithValueLike,
              clang::StringRef, MethodName) {
  auto RT = pointerLikeReturnType(Node);
  if (RT.isNull())
    return false;
  return valueLikeReturnType(findReturnType(Node, MethodName)) == RT;
}

AST_MATCHER(clang::CXXRecordDecl, smartPointerClassWithGetOrValue) {
  auto RT = pointerLikeReturnType(Node);
  if (RT.isNull())
    return false;
  return getLikeReturnType(findReturnType(Node, "get")) == RT ||
         valueLikeReturnType(findReturnType(Node, "value")) == RT;
}

AST_MATCHER(clang::CXXRecordDecl, pointerClass) {
  return !pointerLikeReturnType(Node).isNull();
}

} // namespace

namespace clang::dataflow {

ast_matchers::StatementMatcher isSmartPointerLikeOperatorStar() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("*"),
      callee(cxxMethodDecl(parameterCountIs(0),
                           returns(hasCanonicalType(referenceType())),
                           ofClass(smartPointerClassWithGetOrValue()))));
}

ast_matchers::StatementMatcher isSmartPointerLikeOperatorArrow() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("->"),
      callee(cxxMethodDecl(parameterCountIs(0),
                           returns(hasCanonicalType(pointerType())),
                           ofClass(smartPointerClassWithGetOrValue()))));
}

ast_matchers::StatementMatcher isPointerLikeOperatorStar() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("*"),
      callee(cxxMethodDecl(parameterCountIs(0),
                           returns(hasCanonicalType(referenceType())),
                           ofClass(pointerClass()))));
}

ast_matchers::StatementMatcher isPointerLikeOperatorArrow() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("->"),
      callee(cxxMethodDecl(parameterCountIs(0),
                           returns(hasCanonicalType(pointerType())),
                           ofClass(pointerClass()))));
}

ast_matchers::StatementMatcher
isSmartPointerLikeValueMethodCall(clang::StringRef MethodName) {
  return cxxMemberCallExpr(callee(cxxMethodDecl(
      parameterCountIs(0), returns(hasCanonicalType(referenceType())),
      hasName(MethodName),
      ofClass(smartPointerClassWithValueLike(MethodName)))));
}

ast_matchers::StatementMatcher
isSmartPointerLikeGetMethodCall(clang::StringRef MethodName) {
  return cxxMemberCallExpr(callee(cxxMethodDecl(
      parameterCountIs(0), returns(hasCanonicalType(pointerType())),
      hasName(MethodName), ofClass(smartPointerClassWithGetLike(MethodName)))));
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
