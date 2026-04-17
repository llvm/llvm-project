#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_CHECKUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_CHECKUTILS_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::utils {

/// Emits a configuration diagnostic when a deprecated check alias is enabled
/// and the canonical check name is not also enabled.
inline void diagDeprecatedCheckAlias(ClangTidyCheck &Check,
                                     const ClangTidyContext &Context,
                                     StringRef DeprecatedName,
                                     StringRef CanonicalName) {
  if (!Context.isCheckEnabled(DeprecatedName) ||
      Context.isCheckEnabled(CanonicalName))
    return;

  Check.configurationDiag(
      "'%0' check is deprecated and will be removed in a future release; "
      "consider using '%1' instead")
      << DeprecatedName << CanonicalName;
}

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_CHECKUTILS_H
