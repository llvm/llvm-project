//===--- AvoidNSErrorInitCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_AVOIDNSERRORINITCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_AVOIDNSERRORINITCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace objc {

/// Finds usages of [NSSError init]. It is not the proper way of creating
/// NSError. errorWithDomain:code:userInfo: should be used instead.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/objc-avoid-nserror-init.html
class AvoidNSErrorInitCheck : public ClangTidyCheck {
 public:
  AvoidNSErrorInitCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

}  // namespace objc
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_AVOIDNSERRORINITCHECK_H
