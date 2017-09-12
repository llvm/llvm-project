//===--- OwningMemoryCheck.h - clang-tidy------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_OWNING_MEMORY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_OWNING_MEMORY_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// Checks for common use cases for gsl::owner and enforces the unique owner
/// nature of it whenever possible.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-owning-memory.html
class OwningMemoryCheck : public ClangTidyCheck {
public:
  OwningMemoryCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  bool handleDeletion(const ast_matchers::BoundNodes &Nodes);
  bool handleExpectedOwner(const ast_matchers::BoundNodes &Nodes);
  bool handleAssignmentAndInit(const ast_matchers::BoundNodes &Nodes);
  bool handleAssignmentFromNewOwner(const ast_matchers::BoundNodes &Nodes);
  bool handleReturnValues(const ast_matchers::BoundNodes &Nodes);
  bool handleOwnerMembers(const ast_matchers::BoundNodes &Nodes);
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_OWNING_MEMORY_H
