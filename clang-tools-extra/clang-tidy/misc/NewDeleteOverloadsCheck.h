//===--- NewDeleteOverloadsCheck.h - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NEWDELETEOVERLOADS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NEWDELETEOVERLOADS_H

#include "../ClangTidy.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

namespace clang {
namespace tidy {
namespace misc {

class NewDeleteOverloadsCheck : public ClangTidyCheck {
  std::map<const clang::CXXRecordDecl *,
           llvm::SmallVector<const clang::FunctionDecl *, 4>>
      Overloads;

public:
  NewDeleteOverloadsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NEWDELETEOVERLOADS_H
