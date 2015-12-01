//===--- StaticObjectExceptionCheck.h - clang-tidy---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_ERR58_CPP_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_ERR58_CPP_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

/// Checks whether the constructor for a static or thread_local object will
/// throw.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-static-object-exception.html
class StaticObjectExceptionCheck : public ClangTidyCheck {
public:
  StaticObjectExceptionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_ERR58_CPP_H

