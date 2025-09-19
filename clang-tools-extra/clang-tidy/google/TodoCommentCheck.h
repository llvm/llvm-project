//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::google::readability {

/// Finds TODO comments without a username or bug number.
///
/// Corresponding cpplint.py check: 'readability/todo'
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/readability-todo.html
class TodoCommentCheck : public ClangTidyCheck {
public:
  TodoCommentCheck(StringRef Name, ClangTidyContext *Context);
  ~TodoCommentCheck();

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;

private:
  class TodoCommentHandler;
  std::unique_ptr<TodoCommentHandler> Handler;
};

} // namespace clang::tidy::google::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H
