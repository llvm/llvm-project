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

namespace clang {
namespace tidy {
namespace utils {

/// Finds and fixes header guards.
class HeaderGuardCheck : public ClangTidyCheck {
public:
  HeaderGuardCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerPPCallbacks(CompilerInstance &Compiler) override;

  /// \brief Returns true if the checker should suggest inserting a trailing
  /// comment on the #endif of the header guard. It will use the same name as
  /// returned by getHeaderGuard.
  virtual bool shouldSuggestEndifComment(StringRef Filename);
  /// \brief Returns true if the checker should suggest changing an existing
  /// header guard to the string returned by getHeaderGuard.
  virtual bool shouldFixHeaderGuard(StringRef Filename);
  /// \brief Returns true if the checker should add a header guard to the file
  /// if it has none.
  virtual bool shouldSuggestToAddHeaderGuard(StringRef Filename);
  /// \brief Returns a replacement for endif line with a comment mentioning
  /// \p HeaderGuard. The replacement should start with "endif".
  virtual std::string formatEndIf(StringRef HeaderGuard);
  /// \brief Get the canonical header guard for a file.
  virtual std::string getHeaderGuard(StringRef Filename,
                                     StringRef OldGuard = StringRef()) = 0;
};

} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_HEADERGUARD_H
