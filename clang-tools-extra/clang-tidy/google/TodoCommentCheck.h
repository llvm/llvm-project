//===--- TodoCommentCheck.h - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace readability {

/// Finds TODO comments without a username or bug number.
///
/// Corresponding cpplint.py check: 'readability/todo'
class TodoCommentCheck : public ClangTidyCheck {
public:
  TodoCommentCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;

private:
  class TodoCommentHandler;
  std::unique_ptr<TodoCommentHandler> Handler;
};

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_TODOCOMMENTCHECK_H
