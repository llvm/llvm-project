//===--- ImplicitCastInLoopCheck.h - clang-tidy------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_IMPLICIT_CAST_IN_LOOP_CHECK_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_IMPLICIT_CAST_IN_LOOP_CHECK_H_

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace performance {

// Checks that in a for range loop, if the provided type is a reference, then
// the underlying type is the one returned by the iterator (i.e. that there
// isn't any implicit conversion).
class ImplicitCastInLoopCheck : public ClangTidyCheck {
 public:
   ImplicitCastInLoopCheck(StringRef Name, ClangTidyContext *Context)
       : ClangTidyCheck(Name, Context) {}
   void registerMatchers(ast_matchers::MatchFinder *Finder) override;
   void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

 private:
  void ReportAndFix(const ASTContext *Context, const VarDecl *VD,
                    const CXXOperatorCallExpr *OperatorCall);
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_IMPLICIT_CAST_IN_LOOP_CHECK_H_
