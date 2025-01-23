#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

class MakeFunctionToDirectCheck : public ClangTidyCheck {
public:
  MakeFunctionToDirectCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  // Helper to check if the call is a make_xxx function
  bool isMakeFunction(const std::string &FuncName) const;
  // Get the template type from make_xxx call
  std::string getTemplateType(const CXXConstructExpr *Construct) const;
};

} // namespace clang::tidy::modernize