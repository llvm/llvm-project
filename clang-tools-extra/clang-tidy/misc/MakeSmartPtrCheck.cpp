//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MakeSmartPtrCheck.h"

// FixItHint - Hint to check documentation script to mark this check as
// providing a FixIt.

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

MakeSmartPtrCheck::MakeSmartPtrCheck(StringRef Name, ClangTidyContext *Context)
    : modernize::MakeSmartPtrCheck(Name, Context, "", "") {}

MakeSmartPtrCheck::SmartPtrTypeMatcher
MakeSmartPtrCheck::getSmartPointerTypeMatcher() const {
  return qualType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(classTemplateSpecializationDecl(
          hasName(MakeSmartPtrType), templateArgumentCountIs(1),
          hasTemplateArgument(0, templateArgument(refersToType(
                                     qualType().bind(PointerType)))))))));
}

bool MakeSmartPtrCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  // This check requires both MakeSmartPtrType and MakeSmartPtrFunction to be
  // configured. If either is empty (the default), disable the check.
  if (MakeSmartPtrType.empty())
    return false;
  return LangOpts.CPlusPlus11;
}

} // namespace clang::tidy::misc
