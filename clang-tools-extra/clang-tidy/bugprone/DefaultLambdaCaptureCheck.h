#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DEFAULT_LAMBDA_CAPTURE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DEFAULT_LAMBDA_CAPTURE_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/** Flags lambdas that use default capture modes
 *
 * For the user-facing documentation see:
 * http://clang.llvm.org/extra/clang-tidy/checks/bugprone/default-lambda-capture.html
 */
class DefaultLambdaCaptureCheck : public ClangTidyCheck {
public:
  DefaultLambdaCaptureCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DEFAULT_LAMBDA_CAPTURE_H
