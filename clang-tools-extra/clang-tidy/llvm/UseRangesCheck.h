//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_USERANGESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_USERANGESCHECK_H

#include "../utils/UseRangesCheck.h"

namespace clang::tidy::llvm_check {

/// Finds calls to STL iterator algorithms that can be replaced with LLVM
/// range-based algorithms from `llvm/ADT/STLExtras.h`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvm/use-ranges.html
class UseRangesCheck : public utils::UseRangesCheck {
public:
  UseRangesCheck(StringRef Name, ClangTidyContext *Context);

  ReplacerMap getReplacerMap() const override;
  DiagnosticBuilder createDiag(const CallExpr &Call) override;
  ArrayRef<std::pair<StringRef, StringRef>>
  getFreeBeginEndMethods() const override;
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_USERANGESCHECK_H
