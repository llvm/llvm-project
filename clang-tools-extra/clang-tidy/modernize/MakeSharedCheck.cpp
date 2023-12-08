//===--- MakeSharedCheck.cpp - clang-tidy----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MakeSharedCheck.h"

// FixItHint - Hint to check documentation script to mark this check as
// providing a FixIt.

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

MakeSharedCheck::MakeSharedCheck(StringRef Name, ClangTidyContext *Context)
    : MakeSmartPtrCheck(Name, Context, "std::make_shared") {}

MakeSharedCheck::SmartPtrTypeMatcher
MakeSharedCheck::getSmartPointerTypeMatcher() const {
  return qualType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(classTemplateSpecializationDecl(
          hasName("::std::shared_ptr"), templateArgumentCountIs(1),
          hasTemplateArgument(0, templateArgument(refersToType(
                                     qualType().bind(PointerType)))))))));
}

} // namespace clang::tidy::modernize
