//===--- ReservedIdentifierCheck.h - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RESERVEDIDENTIFIERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RESERVEDIDENTIFIERCHECK_H

#include "../utils/RenamerClangTidyCheck.h"
#include "llvm/ADT/Optional.h"
#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace bugprone {

/// Checks for usages of identifiers reserved for use by the implementation.
///
/// The C and C++ standards both reserve the following names for such use:
/// * identifiers that begin with an underscore followed by an uppercase letter;
/// * identifiers in the global namespace that begin with an underscore.
///
/// The C standard additionally reserves names beginning with a double
/// underscore, while the C++ standard strengthens this to reserve names with a
/// double underscore occurring anywhere.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-reserved-identifier.html
class ReservedIdentifierCheck final : public RenamerClangTidyCheck {
  const bool Invert;
  const std::vector<StringRef> AllowedIdentifiers;

public:
  ReservedIdentifierCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  llvm::Optional<FailureInfo>
  getDeclFailureInfo(const NamedDecl *Decl,
                     const SourceManager &SM) const override;
  llvm::Optional<FailureInfo>
  getMacroFailureInfo(const Token &MacroNameTok,
                      const SourceManager &SM) const override;
  DiagInfo getDiagInfo(const NamingCheckId &ID,
                       const NamingCheckFailure &Failure) const override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_RESERVEDIDENTIFIERCHECK_H
