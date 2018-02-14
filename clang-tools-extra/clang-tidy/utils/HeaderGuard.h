//===--- HeaderGuard.h - clang-tidy -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_HEADERGUARD_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_HEADERGUARD_H

#include "../ClangTidy.h"
#include "../utils/HeaderFileExtensionsUtils.h"

namespace clang {
namespace tidy {
namespace utils {

/// Finds and fixes header guards.
/// The check supports these options:
///   - `HeaderFileExtensions`: a comma-separated list of filename extensions of
///     header files (The filename extension should not contain "." prefix).
///     ",h,hh,hpp,hxx" by default.
///     For extension-less header files, using an empty string or leaving an
///     empty string between "," if there are other filename extensions.
class HeaderGuardCheck : public ClangTidyCheck {
public:
  HeaderGuardCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        RawStringHeaderFileExtensions(Options.getLocalOrGlobal(
            "HeaderFileExtensions", utils::defaultHeaderFileExtensions())) {
    utils::parseHeaderFileExtensions(RawStringHeaderFileExtensions,
                                     HeaderFileExtensions, ',');
  }
  void registerPPCallbacks(CompilerInstance &Compiler) override;

  /// Returns ``true`` if the check should suggest inserting a trailing comment
  /// on the ``#endif`` of the header guard. It will use the same name as
  /// returned by ``HeaderGuardCheck::getHeaderGuard``.
  virtual bool shouldSuggestEndifComment(StringRef Filename);
  /// Returns ``true`` if the check should suggest changing an existing header
  /// guard to the string returned by ``HeaderGuardCheck::getHeaderGuard``.
  virtual bool shouldFixHeaderGuard(StringRef Filename);
  /// Returns ``true`` if the check should add a header guard to the file
  /// if it has none.
  virtual bool shouldSuggestToAddHeaderGuard(StringRef Filename);
  /// Returns a replacement for the ``#endif`` line with a comment mentioning
  /// \p HeaderGuard. The replacement should start with ``endif``.
  virtual std::string formatEndIf(StringRef HeaderGuard);
  /// Gets the canonical header guard for a file.
  virtual std::string getHeaderGuard(StringRef Filename,
                                     StringRef OldGuard = StringRef()) = 0;

private:
  std::string RawStringHeaderFileExtensions;
  utils::HeaderFileExtensionsSet HeaderFileExtensions;
};

} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_HEADERGUARD_H
