//===--- MakeSharedCheck.cpp - clang-tidy----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MakeSharedCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

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

} // namespace modernize
} // namespace tidy
} // namespace clang
