//===--- SharedPtrArrayMismatchCheck.h - clang-tidy -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SHAREDPTRARRAYMISMATCHCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SHAREDPTRARRAYMISMATCHCHECK_H

#include "SmartPtrArrayMismatchCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Find `std::shared_ptr<T>(new T[...])`, replace it (if applicable) with
/// `std::shared_ptr<T[]>(new T[...])`.
///
/// Example:
///
/// \code
///   std::shared_ptr<int> PtrArr{new int[10]};
/// \endcode
class SharedPtrArrayMismatchCheck : public SmartPtrArrayMismatchCheck {
public:
  SharedPtrArrayMismatchCheck(StringRef Name, ClangTidyContext *Context);

protected:
  SmartPtrClassMatcher getSmartPointerClassMatcher() const override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SHAREDPTRARRAYMISMATCHCHECK_H
