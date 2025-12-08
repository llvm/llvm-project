#ifndef DEVTOOLS_CYMBAL_CLANG_TIDY_RUNTIME_UPSTREAM_FRAMEWORK_UNCHECKED_STATUSOR_ACCESS_H_
#define DEVTOOLS_CYMBAL_CLANG_TIDY_RUNTIME_UPSTREAM_FRAMEWORK_UNCHECKED_STATUSOR_ACCESS_H_

#include <optional>

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang::tidy::abseil {

// Warns when the code is unwrapping an absl::StatusOr<T> object without
// assuring that it contains a value.
//
// For details on the dataflow analysis implemented in this check see:
// http://google3/devtools/cymbal/nullability/statusor
class UncheckedStatusOrAccessCheck : public ClangTidyCheck {
 public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
};

}  // namespace clang::abseil

#endif  // DEVTOOLS_CYMBAL_CLANG_TIDY_RUNTIME_UPSTREAM_FRAMEWORK_UNCHECKED_STATUSOR_ACCESS_H_
