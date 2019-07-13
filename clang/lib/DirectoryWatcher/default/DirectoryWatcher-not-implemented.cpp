//===- DirectoryWatcher-not-implemented.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DirectoryWatcher/DirectoryWatcher.h"

using namespace llvm;
using namespace clang;

std::unique_ptr<DirectoryWatcher> clang::DirectoryWatcher::create(
    StringRef Path,
    std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)> Receiver,
    bool WaitForInitialSync) {
  return nullptr;
}