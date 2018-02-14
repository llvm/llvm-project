//===--- ForbiddenSubclassingCheck.h - clang-tidy ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_FORBIDDEN_SUBCLASSING_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_FORBIDDEN_SUBCLASSING_CHECK_H

#include "../ClangTidy.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace objc {

/// Finds Objective-C classes which have a superclass which is
/// documented to not support subclassing.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/objc-forbidden-subclassing.html
class ForbiddenSubclassingCheck : public ClangTidyCheck {
public:
  ForbiddenSubclassingCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;

private:
  const std::vector<std::string> ForbiddenSuperClassNames;
};

} // namespace objc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_FORBIDDEN_SUBCLASSING_CHECK_H
