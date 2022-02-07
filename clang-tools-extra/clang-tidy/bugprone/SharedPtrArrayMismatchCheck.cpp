//===--- SharedPtrArrayMismatchCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SharedPtrArrayMismatchCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

SharedPtrArrayMismatchCheck::SharedPtrArrayMismatchCheck(
    StringRef Name, ClangTidyContext *Context)
    : SmartPtrArrayMismatchCheck(Name, Context, "shared") {}

SharedPtrArrayMismatchCheck::SmartPtrClassMatcher
SharedPtrArrayMismatchCheck::getSmartPointerClassMatcher() const {
  return classTemplateSpecializationDecl(
      hasName("::std::shared_ptr"), templateArgumentCountIs(1),
      hasTemplateArgument(
          0, templateArgument(refersToType(qualType().bind(PointerTypeN)))));
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
