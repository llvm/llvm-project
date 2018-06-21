//===--- ForwardingReferenceOverloadCheck.h - clang-tidy---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FORWARDINGREFERENCEOVERLOADCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FORWARDINGREFERENCEOVERLOADCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// The checker looks for constructors that can act as copy or move constructors
/// through their forwarding reference parameters. If a non const lvalue
/// reference is passed to the constructor, the forwarding reference parameter
/// can be a perfect match while the const reference parameter of the copy
/// constructor can't. The forwarding reference constructor will be called,
/// which can lead to confusion.
/// For detailed description of this problem see: Scott Meyers, Effective Modern
/// C++ Design, item 26.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-forwarding-reference-overload.html
class ForwardingReferenceOverloadCheck : public ClangTidyCheck {
public:
  ForwardingReferenceOverloadCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FORWARDINGREFERENCEOVERLOADCHECK_H
