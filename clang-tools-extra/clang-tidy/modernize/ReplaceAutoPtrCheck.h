//===--- ReplaceAutoPtrCheck.h - clang-tidy----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_AUTO_PTR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_AUTO_PTR_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Transforms the deprecated `std::auto_ptr` into the C++11 `std::unique_ptr`.
///
/// Note that both the `std::auto_ptr` type and the transfer of ownership are
/// transformed. `std::auto_ptr` provides two ways to transfer the ownership,
/// the copy-constructor and the assignment operator. Unlike most classes these
/// operations do not 'copy' the resource but they 'steal' it.
/// `std::unique_ptr` uses move semantics instead, which makes the intent of
/// transferring the resource explicit. This difference between the two smart
/// pointers requeres to wrap the copy-ctor and assign-operator with
/// `std::move()`.
///
/// For example, given:
///
/// \code
///   std::auto_ptr<int> i, j;
///   i = j;
/// \endcode
///
/// This code is transformed to:
///
/// \code
///   std::unique_ptr<in> i, j;
///   i = std::move(j);
/// \endcode
class ReplaceAutoPtrCheck : public ClangTidyCheck {
public:
  ReplaceAutoPtrCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  utils::IncludeInserter Inserter;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_AUTO_PTR_H
