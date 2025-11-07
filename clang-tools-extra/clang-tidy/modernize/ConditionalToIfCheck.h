#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CONDITIONALTOIFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CONDITIONALTOIFCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Convert between simple conditional (?:) expressions and equivalent if/else
/// statements in safe, syntactically simple cases.
///
/// Direction is controlled by the option:
///   - modernize-conditional-to-if.PreferredForm: "if" | "conditional"
class ConditionalToIfCheck : public ClangTidyCheck {
public:
  ConditionalToIfCheck(llvm::StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  enum class Preferred { If, Conditional };
  Preferred PreferredForm;

  // Conservative side-effect check (needs ASTContext).
  static bool hasObviousSideEffects(const Expr *E, ASTContext &Ctx);

  // Fetch exact source text for a range.
  static std::string getText(const SourceRange &R,
                             const ast_matchers::MatchFinder::MatchResult &Rst);

  // Ensure locations are in main file and not in macros.
  static bool locationsAreOK(const SourceRange &R,
                             const ast_matchers::MatchFinder::MatchResult &Rst);
};

} // namespace clang::tidy::modernize

#endif
