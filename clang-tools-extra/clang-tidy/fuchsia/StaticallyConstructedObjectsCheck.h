//===--- StaticallyConstructedObjectsCheck.h - clang-tidy--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_STATICALLY_CONSTRUCTED_OBJECTS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_STATICALLY_CONSTRUCTED_OBJECTS_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace fuchsia {

/// Constructing global, non-trivial objects with static storage is
/// disallowed, unless the object is statically initialized with a constexpr 
/// constructor or has no explicit constructor.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-statically-constructed-objects.html
class StaticallyConstructedObjectsCheck : public ClangTidyCheck {
public:
  StaticallyConstructedObjectsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace fuchsia
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_STATICALLY_CONSTRUCTED_OBJECTS_H
