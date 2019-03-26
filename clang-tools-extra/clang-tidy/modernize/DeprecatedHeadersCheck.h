//===--- DeprecatedHeadersCheck.h - clang-tidy-------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// This check replaces deprecated C library headers with their C++ STL
/// alternatives.
///
/// Before:
/// ~~~{.cpp}
/// #include <header.h>
/// ~~~
///
/// After:
/// ~~~{.cpp}
/// #include <cheader>
/// ~~~
///
/// Example: ``<stdio.h> => <cstdio>``
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-deprecated-headers.html
class DeprecatedHeadersCheck : public ClangTidyCheck {
public:
  DeprecatedHeadersCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H
