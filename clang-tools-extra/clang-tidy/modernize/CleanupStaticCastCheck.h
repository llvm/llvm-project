// CleanupStaticCastCheck.h
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CLEANUPSTATICCASTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CLEANUPSTATICCASTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds and removes static_cast where target type exactly matches source type.
///
/// This check helps clean up redundant static_cast operations that remain after
/// type system changes, improving code readability and maintainability.
///
/// For the given code:
/// \code
///   size_t s = 42;
///   foo(static_cast<size_t>(s));
/// \endcode
///
/// The check will suggest removing the redundant cast:
/// \code
///   size_t s = 42;
///   foo(s);
/// \endcode
///
/// Note: This check intentionally ignores redundant casts in template instantiations
/// as they might be needed for other template parameter types.
class CleanupStaticCastCheck : public ClangTidyCheck {
public:
  CleanupStaticCastCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CLEANUPSTATICCASTCHECK_H