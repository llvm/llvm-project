//===--- StringviewNullptrCheck.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STRINGVIEWNULLPTRCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STRINGVIEWNULLPTRCHECK_H

#include "../utils/TransformerClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Checks for various ways that the `const CharT*` constructor of
/// `std::basic_string_view` can be passed a null argument and replaces them
/// with the default constructor in most cases. For the comparison operators,
/// braced initializer list does not compile so instead a call to `.empty()` or
/// the empty string literal are used, where appropriate.
///
/// This prevents code from invoking behavior which is unconditionally
/// undefined. The single-argument `const CharT*` constructor does not check
/// for the null case before dereferencing its input. The standard is slated to
/// add an explicitly-deleted overload to catch some of these cases:
/// wg21.link/p2166
///
/// To catch the additional cases of `NULL` (which expands to `__null`) and
/// `0`, first run the ``modernize-use-nullptr`` check to convert the callers
/// to `nullptr`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/stringview-nullptr.html
class StringviewNullptrCheck : public utils::TransformerClangTidyCheck {
public:
  StringviewNullptrCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STRINGVIEWNULLPTRCHECK_H
