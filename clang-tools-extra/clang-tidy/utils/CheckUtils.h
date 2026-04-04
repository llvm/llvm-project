#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_CHECKUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_CHECKUTILS_H

#include "../ClangTidyDiagnosticConsumer.h"

namespace clang::tidy::utils {

/// Returns true when a check is running under a deprecated name and the user
/// should be prompted to migrate to CanonicalName.
inline bool isDeprecatedAlias(const ClangTidyContext &Context,
                              StringRef DeprecatedName,
                              StringRef CanonicalName) {
  return Context.isCheckEnabled(DeprecatedName) &&
         !Context.isCheckEnabled(CanonicalName);
}

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_CHECKUTILS_H
