//===--- MakeUniqueCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_UNIQUE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_UNIQUE_H

#include "MakeSmartPtrCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Replace the pattern:
/// \code
///   std::unique_ptr<type>(new type(args...))
/// \endcode
///
/// With the C++14 version:
/// \code
///   std::make_unique<type>(args...)
/// \endcode
class MakeUniqueCheck : public MakeSmartPtrCheck {
public:
  MakeUniqueCheck(StringRef Name, ClangTidyContext *Context);

protected:
  SmartPtrTypeMatcher getSmartPointerTypeMatcher() const override;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

private:
  const bool RequireCPlusPlus14;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_UNIQUE_H
