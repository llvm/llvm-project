//===--- NonConstReferences.h - clang-tidy ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NON_CONST_REFERENCES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NON_CONST_REFERENCES_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace runtime {

/// \brief Checks the usage of non-constant references in function parameters.
///
/// https://google.github.io/styleguide/cppguide.html#Reference_Arguments
class NonConstReferences : public ClangTidyCheck {
public:
  NonConstReferences(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::vector<std::string> WhiteListTypes;
};

} // namespace runtime
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NON_CONST_REFERENCES_H
