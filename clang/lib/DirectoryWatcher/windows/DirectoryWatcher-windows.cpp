//===- DirectoryWatcher-windows.cpp - Windows-platform directory watching -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: This is not yet an implementation, but it will make it so Windows
//       builds don't fail.

#include "DirectoryScanner.h"
#include "clang/DirectoryWatcher/DirectoryWatcher.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Path.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace llvm;
using namespace clang;

class DirectoryWatcherWindows : public clang::DirectoryWatcher {
public:
  ~DirectoryWatcherWindows() override { }
  void InitialScan() { }
  void EventReceivingLoop() { }
  void StopWork() { }
};
} // namespace

llvm::Expected<std::unique_ptr<DirectoryWatcher>>
clang::DirectoryWatcher::create(
    StringRef Path,
    std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)> Receiver,
    bool WaitForInitialSync) {
  return llvm::Expected<std::unique_ptr<DirectoryWatcher>>(
      llvm::errorCodeToError(std::make_error_code(std::errc::not_supported)));
}
