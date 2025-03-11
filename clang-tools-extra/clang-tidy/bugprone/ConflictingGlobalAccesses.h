//===--- ConflictingGlobalAccesses.h - clang-tidy ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIDEEFFECTBETWEENSEQUENCE\
POINTSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIDEEFFECTBETWEENSEQUENCE\
POINTSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds conflicting accesses on global variables.
///
/// Modifying twice or reading and modifying a memory location without a
/// defined sequence of the operations is undefined behavior. This checker is
/// similar to the -Wunsequenced clang warning, however it only looks at global
/// variables and can find unsequenced operations inside functions as well.
///
/// For example: \code
///
/// int a = 0;
/// int b = (a++) - a; // This is flagged by -Wunsequenced.
///
/// \endcode
///
/// However global variables allow for more complex scenarios that
/// -Wunsequenced doesn't detect. E.g. \code
///
/// int globalVar = 0;
///
/// int incFun() {
///   globalVar++;
///   return globalVar;
/// }
///
/// int main() {
///   return globalVar + incFun(); // This is not detected by -Wunsequenced.
/// }
///
/// \endcode
///
/// This checker attempts to detect such undefined behavior. It recurses into
/// functions that are inside the same translation unit. It also attempts not to
/// flag cases that are already covered by -Wunsequenced. Global unions and
/// structs are also handled.
class ConflictingGlobalAccesses : public ClangTidyCheck {
public:
  ConflictingGlobalAccesses(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif
