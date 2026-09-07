//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/BugReporter.h"

llvm::ArrayRef<llvm::StringRef> lldb_private::GetBugReportQuestions() {
  static constexpr llvm::StringRef g_questions[] = {
      "What were you doing?",
      "What did you expect to happen?",
      "What happened instead?",
  };
  return g_questions;
}
