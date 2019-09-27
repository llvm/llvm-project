//===--- DispatchOnceNonstaticCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DispatchOnceNonstaticCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace darwin {

void DispatchOnceNonstaticCheck::registerMatchers(MatchFinder *Finder) {
  // Find variables without static or global storage. VarDecls do not include
  // struct/class members, which are FieldDecls.
  Finder->addMatcher(
      varDecl(hasLocalStorage(), hasType(asString("dispatch_once_t")))
          .bind("non-static-var"),
      this);

  // Members of structs or classes might be okay, if the use is at static or
  // global scope. These will be ignored for now. But ObjC ivars can be
  // flagged immediately, since they cannot be static.
  Finder->addMatcher(
      objcIvarDecl(hasType(asString("dispatch_once_t"))).bind("ivar"), this);
}

void DispatchOnceNonstaticCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("non-static-var")) {
    if (const auto *PD = dyn_cast<ParmVarDecl>(VD)) {
      // Catch function/method parameters, as any dispatch_once_t should be
      // passed by pointer instead.
      diag(PD->getTypeSpecStartLoc(),
           "dispatch_once_t variables must have static or global storage "
           "duration; function parameters should be pointer references");
    } else {
      diag(VD->getTypeSpecStartLoc(), "dispatch_once_t variables must have "
                                      "static or global storage duration")
          << FixItHint::CreateInsertion(VD->getTypeSpecStartLoc(), "static ");
    }
  }

  if (const auto *D = Result.Nodes.getNodeAs<ObjCIvarDecl>("ivar")) {
    diag(D->getTypeSpecStartLoc(),
         "dispatch_once_t variables must have static or global storage "
         "duration and cannot be Objective-C instance variables");
  }
}

} // namespace darwin
} // namespace tidy
} // namespace clang
