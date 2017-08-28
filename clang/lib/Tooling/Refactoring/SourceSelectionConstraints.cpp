//===--- SourceSelectionConstraints.cpp - Evaluate selection constraints --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/SourceSelectionConstraints.h"
#include "clang/Tooling/Refactoring/RefactoringRuleContext.h"

using namespace clang;
using namespace tooling;
using namespace selection;

Optional<SourceSelectionRange>
SourceSelectionRange::evaluate(RefactoringRuleContext &Context) {
  SourceRange R = Context.getSelectionRange();
  if (R.isInvalid())
    return None;
  return SourceSelectionRange(Context.getSources(), R);
}
