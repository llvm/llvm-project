//===--- Check.h - clangd check fuction ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions and structure definitions for Check.cpp
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "llvm/ADT/StringRef.h"
#include <string>

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TOOL_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TOOL_CHECK_H

namespace clang {
namespace clangd {

struct ClangdCheckOptions {
  std::optional<llvm::StringRef> CheckTidyTime;
  llvm::StringRef CheckFileLines;
  bool CheckLocations;
  bool CheckCompletion;
  bool CheckWarnings;
};

bool check(llvm::StringRef File, const ThreadsafeFS &TFS,
           const ClangdLSPServer::Options &Opts,
           const ClangdCheckOptions &CheckOpts);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_TOOL_CHECK_H
