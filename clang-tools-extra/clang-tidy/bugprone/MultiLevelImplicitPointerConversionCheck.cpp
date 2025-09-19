//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MultiLevelImplicitPointerConversionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static unsigned getPointerLevel(const QualType &PtrType) {
  if (!PtrType->isPointerType())
    return 0U;

  return 1U + getPointerLevel(PtrType->castAs<PointerType>()->getPointeeType());
}

namespace {

AST_MATCHER(ImplicitCastExpr, isMultiLevelPointerConversion) {
  const QualType TargetType = Node.getType()
                                  .getCanonicalType()
                                  .getNonReferenceType()
                                  .getUnqualifiedType();
  const QualType SourceType = Node.getSubExpr()
                                  ->getType()
                                  .getCanonicalType()
                                  .getNonReferenceType()
                                  .getUnqualifiedType();

  if (TargetType == SourceType)
    return false;

  const unsigned TargetPtrLevel = getPointerLevel(TargetType);
  if (0U == TargetPtrLevel)
    return false;

  const unsigned SourcePtrLevel = getPointerLevel(SourceType);
  if (0U == SourcePtrLevel)
    return false;

  return SourcePtrLevel != TargetPtrLevel;
}

AST_MATCHER(QualType, isPointerType) {
  const QualType Type =
      Node.getCanonicalType().getNonReferenceType().getUnqualifiedType();

  return !Type.isNull() && Type->isPointerType();
}

} // namespace

MultiLevelImplicitPointerConversionCheck::
    MultiLevelImplicitPointerConversionCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), EnableInC(Options.get("EnableInC", true)) {
}

void MultiLevelImplicitPointerConversionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "EnableInC", EnableInC);
}

void MultiLevelImplicitPointerConversionCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      implicitCastExpr(hasCastKind(CK_BitCast), isMultiLevelPointerConversion(),
                       unless(hasParent(explicitCastExpr(
                           hasDestinationType(isPointerType())))))
          .bind("expr"),
      this);
}

std::optional<TraversalKind>
MultiLevelImplicitPointerConversionCheck::getCheckTraversalKind() const {
  return TK_AsIs;
}

void MultiLevelImplicitPointerConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<ImplicitCastExpr>("expr");
  QualType Target = MatchedExpr->getType().getDesugaredType(*Result.Context);
  QualType Source =
      MatchedExpr->getSubExpr()->getType().getDesugaredType(*Result.Context);

  diag(MatchedExpr->getExprLoc(), "multilevel pointer conversion from %0 to "
                                  "%1, please use explicit cast")
      << Source << Target;
}

} // namespace clang::tidy::bugprone
