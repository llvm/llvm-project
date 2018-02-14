//===--- AvoidThrowingObjCExceptionCheck.h - clang-tidy----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_AVOID_THROWING_EXCEPTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_AVOID_THROWING_EXCEPTION_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace objc {

/// The check is to find usage of @throw invocation in Objective-C code.
/// We should avoid using @throw for Objective-C exceptions according to
/// the Google Objective-C Style Guide.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google-objc-avoid-throwing-exception.html
class AvoidThrowingObjCExceptionCheck : public ClangTidyCheck {
 public:
  AvoidThrowingObjCExceptionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

}  // namespace objc
}  // namespace google
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_AVOID_THROWING_EXCEPTION_H
