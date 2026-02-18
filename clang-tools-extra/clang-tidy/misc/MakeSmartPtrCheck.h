//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MAKESMARTPTRCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MAKESMARTPTRCHECK_H

#include "../modernize/MakeSmartPtrCheck.h"

namespace clang::tidy::misc {

/// Finds constructions of custom smart pointer types from raw ``new``
/// expressions and replaces them with a configurable factory function.
///
/// Unlike ``modernize-make-shared`` and ``modernize-make-unique``, this check
/// has no default smart pointer type or factory function. Both
/// ``MakeSmartPtrType`` and ``MakeSmartPtrFunction`` must be configured for the
/// check to produce diagnostics.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/misc/make-smart-ptr.html
class MakeSmartPtrCheck : public modernize::MakeSmartPtrCheck {
public:
  MakeSmartPtrCheck(StringRef Name, ClangTidyContext *Context);

protected:
  SmartPtrTypeMatcher getSmartPointerTypeMatcher() const override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MAKESMARTPTRCHECK_H
