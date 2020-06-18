//===--- DeallocInCategoryCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeallocInCategoryCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

void DeallocInCategoryCheck::registerMatchers(MatchFinder *Finder) {
  // Non-NSObject/NSProxy-derived objects may not have -dealloc as a special
  // method. However, it seems highly unrealistic to expect many false-positives
  // by warning on -dealloc in categories on classes without one of those
  // base classes.
  Finder->addMatcher(
      objcMethodDecl(isInstanceMethod(), hasName("dealloc"),
                     hasDeclContext(objcCategoryImplDecl().bind("impl")))
          .bind("dealloc"),
      this);
}

void DeallocInCategoryCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *DeallocDecl = Result.Nodes.getNodeAs<ObjCMethodDecl>("dealloc");
  const auto *CID = Result.Nodes.getNodeAs<ObjCCategoryImplDecl>("impl");
  assert(DeallocDecl != nullptr);
  diag(DeallocDecl->getLocation(), "category %0 should not implement -dealloc")
      << CID;
}

} // namespace objc
} // namespace tidy
} // namespace clang
