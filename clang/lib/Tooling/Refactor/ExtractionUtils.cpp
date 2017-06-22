//===--- ExtractionUtils.cpp - Extraction helper functions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExtractionUtils.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"

using namespace clang;

Optional<StringRef> tooling::extract::nameForExtractedVariable(const Expr *E) {
  if (const auto *Call = dyn_cast<CallExpr>(E)) {
    if (const auto *Fn = Call->getDirectCallee())
      return Fn->getName();
  } else if (const auto *Msg = dyn_cast<ObjCMessageExpr>(E)) {
    if (const auto *M = Msg->getMethodDecl()) {
      if (M->getSelector().isUnarySelector())
        return M->getSelector().getNameForSlot(0);
    }
  } else if (const auto *PRE = dyn_cast<ObjCPropertyRefExpr>(E)) {
    if (PRE->isImplicitProperty()) {
      if (const auto *M = PRE->getImplicitPropertyGetter())
        return M->getSelector().getNameForSlot(0);
    } else if (const auto *Prop = PRE->getExplicitProperty())
      return Prop->getName();
  }
  return None;
}
