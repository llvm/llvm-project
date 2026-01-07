//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MakeUniqueCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

MakeUniqueCheck::MakeUniqueCheck(StringRef Name,
                                 clang::tidy::ClangTidyContext *Context)
    : MakeSmartPtrCheck(Name, Context, "std::make_unique"),
      RequireCPlusPlus14(Options.get("MakeSmartPtrFunction", "").empty()) {}

MakeUniqueCheck::SmartPtrTypeMatcher
MakeUniqueCheck::getSmartPointerTypeMatcher() const {
  return qualType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(classTemplateSpecializationDecl(
          hasName("::std::unique_ptr"), templateArgumentCountIs(2),
          hasTemplateArgument(
              0, templateArgument(refersToType(qualType().bind(PointerType)))),
          hasTemplateArgument(
              1, templateArgument(refersToType(
                     qualType(hasDeclaration(classTemplateSpecializationDecl(
                         hasName("::std::default_delete"),
                         templateArgumentCountIs(1),
                         hasTemplateArgument(
                             0, templateArgument(refersToType(qualType(
                                    equalsBoundNode(PointerType))))))))))))))));
}

bool MakeUniqueCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return RequireCPlusPlus14 ? LangOpts.CPlusPlus14 : LangOpts.CPlusPlus11;
}

// FixItHint is done by MakeSmartPtrCheck

} // namespace clang::tidy::modernize
