//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"

namespace libcpp {
class proper_version_checks : public clang::tidy::ClangTidyCheck {
public:
  proper_version_checks(llvm::StringRef, clang::tidy::ClangTidyContext*);
  void registerPPCallbacks(const clang::SourceManager& source_manager,
                           clang::Preprocessor* preprocessor,
                           clang::Preprocessor* module_expander) override;
};
} // namespace libcpp
