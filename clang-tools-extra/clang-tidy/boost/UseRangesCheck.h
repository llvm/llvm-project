//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USERANGESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USERANGESCHECK_H

#include "../utils/UseRangesCheck.h"

namespace clang::tidy::boost {

/// Detects calls to standard library iterator algorithms that could be
/// replaced with a boost ranges version instead
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/boost/use-ranges.html
class UseRangesCheck : public utils::UseRangesCheck {
public:
  UseRangesCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  ReplacerMap getReplacerMap() const override;

  DiagnosticBuilder createDiag(const CallExpr &Call) override;

  ArrayRef<std::pair<StringRef, StringRef>>
  getFreeBeginEndMethods() const override;

  std::optional<ReverseIteratorDescriptor>
  getReverseDescriptor() const override;

private:
  bool IncludeBoostSystem;
  bool UseReversePipe;
};

} // namespace clang::tidy::boost

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USERANGESCHECK_H
