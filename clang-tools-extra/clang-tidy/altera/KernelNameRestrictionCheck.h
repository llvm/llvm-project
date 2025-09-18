//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_KERNEL_NAME_RESTRICTION_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_KERNEL_NAME_RESTRICTION_CHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::altera {

/// Finds kernel files and include directives whose filename is `kernel.cl`,
/// `Verilog.cl`, or `VHDL.cl`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/altera/kernel-name-restriction.html
class KernelNameRestrictionCheck : public ClangTidyCheck {
public:
  KernelNameRestrictionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *) override;
};

} // namespace clang::tidy::altera

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_KERNEL_NAME_RESTRICTION_CHECK_H
