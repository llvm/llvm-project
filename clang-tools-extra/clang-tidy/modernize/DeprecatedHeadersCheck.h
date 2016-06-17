//===--- DeprecatedHeadersCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H

#include "../ClangTidy.h"

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
  void registerPPCallbacks(CompilerInstance &Compiler) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H
