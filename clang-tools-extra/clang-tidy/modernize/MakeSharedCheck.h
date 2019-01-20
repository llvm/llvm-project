//===--- MakeSharedCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SHARED_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SHARED_H

#include "MakeSmartPtrCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Replace the pattern:
/// \code
///   std::shared_ptr<type>(new type(args...))
/// \endcode
///
/// With the safer version:
/// \code
///   std::make_shared<type>(args...)
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-make-shared.html
class MakeSharedCheck : public MakeSmartPtrCheck {
public:
  MakeSharedCheck(StringRef Name, ClangTidyContext *Context);

protected:
  SmartPtrTypeMatcher getSmartPointerTypeMatcher() const override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SHARED_H
