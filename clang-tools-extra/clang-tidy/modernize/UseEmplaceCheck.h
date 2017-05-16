//===--- UseEmplaceCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EMPLACE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EMPLACE_H

#include "../ClangTidy.h"
#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace modernize {

/// This check looks for cases when inserting new element into std::vector but
/// the element is constructed temporarily.
/// It replaces those calls for emplace_back of arguments passed to
/// constructor of temporary object.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-emplace.html
class UseEmplaceCheck : public ClangTidyCheck {
public:
  UseEmplaceCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  std::vector<std::string> ContainersWithPushBack;
  std::vector<std::string> SmartPointers;
  std::vector<std::string> TupleTypes;
  std::vector<std::string> TupleMakeFunctions;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EMPLACE_H
