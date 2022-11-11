//===--- SuspiciousMemoryComparisonCheck.cpp - clang-tidy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousMemoryComparisonCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

static llvm::Optional<uint64_t> tryEvaluateSizeExpr(const Expr *SizeExpr,
                                                    const ASTContext &Ctx) {
  Expr::EvalResult Result;
  if (SizeExpr->EvaluateAsRValue(Result, Ctx))
    return Ctx.toBits(
        CharUnits::fromQuantity(Result.Val.getInt().getExtValue()));
  return None;
}

void SuspiciousMemoryComparisonCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(allOf(callee(namedDecl(
                         anyOf(hasName("::memcmp"), hasName("::std::memcmp")))),
                     unless(isInstantiationDependent())))
          .bind("call"),
      this);
}

void SuspiciousMemoryComparisonCheck::check(
    const MatchFinder::MatchResult &Result) {
  const ASTContext &Ctx = *Result.Context;
  const auto *CE = Result.Nodes.getNodeAs<CallExpr>("call");

  const Expr *SizeExpr = CE->getArg(2);
  assert(SizeExpr != nullptr && "Third argument of memcmp is mandatory.");
  llvm::Optional<uint64_t> ComparedBits = tryEvaluateSizeExpr(SizeExpr, Ctx);

  for (unsigned int ArgIndex = 0; ArgIndex < 2; ++ArgIndex) {
    const Expr *ArgExpr = CE->getArg(ArgIndex);
    QualType ArgType = ArgExpr->IgnoreImplicit()->getType();
    const Type *PointeeType = ArgType->getPointeeOrArrayElementType();
    assert(PointeeType != nullptr && "PointeeType should always be available.");
    QualType PointeeQualifiedType(PointeeType, 0);

    if (PointeeType->isRecordType()) {
      if (const RecordDecl *RD =
              PointeeType->getAsRecordDecl()->getDefinition()) {
        if (const auto *CXXDecl = dyn_cast<CXXRecordDecl>(RD)) {
          if (!CXXDecl->isStandardLayout()) {
            diag(CE->getBeginLoc(),
                 "comparing object representation of non-standard-layout type "
                 "%0; consider using a comparison operator instead")
                << PointeeQualifiedType;
            break;
          }
        }
      }
    }

    if (!PointeeType->isIncompleteType()) {
      uint64_t PointeeSize = Ctx.getTypeSize(PointeeType);
      if (ComparedBits && *ComparedBits >= PointeeSize &&
          !Ctx.hasUniqueObjectRepresentations(PointeeQualifiedType)) {
        diag(CE->getBeginLoc(),
             "comparing object representation of type %0 which does not have a "
             "unique object representation; consider comparing %select{the "
             "values|the members of the object}1 manually")
            << PointeeQualifiedType << (PointeeType->isRecordType() ? 1 : 0);
        break;
      }
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
