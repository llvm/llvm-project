//===--- LLVMFilesystemAccessCheck.h - clang-tidy ---------------*- C++ -*-===//
//
// Enforces controlled filesystem access patterns
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_IOSANDBOXCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_IOSANDBOXCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Enforces two rules for filesystem access:
/// 1. Functions calling llvm::sys::fs must have a sandbox bypass RAII object
/// 2. Only llvm::sys::fs functions may call low-level filesystem functions
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/llvm/io-sandbox.html
class IOSandboxCheck : public ClangTidyCheck {
public:
  IOSandboxCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_IOSANDBOXCHECK_H
