//===--- Cancellation.cpp -----------------------------------------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Cancellation.h"
#include <atomic>

namespace clang {
namespace clangd {

namespace {
static Key<ConstTaskHandle> TaskKey;
} // namespace

char CancelledError::ID = 0;

const Task &getCurrentTask() {
  const auto TH = Context::current().getExisting(TaskKey);
  assert(TH && "Fetched a nullptr for TaskHandle from context.");
  return *TH;
}

Context setCurrentTask(ConstTaskHandle TH) {
  assert(TH && "Trying to stash a nullptr as TaskHandle into context.");
  return Context::current().derive(TaskKey, std::move(TH));
}

} // namespace clangd
} // namespace clang
