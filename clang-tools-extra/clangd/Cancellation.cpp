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

char CancelledError::ID = 0;
static Key<std::shared_ptr<std::atomic<bool>>> FlagKey;

std::pair<Context, Canceler> cancelableTask() {
  auto Flag = std::make_shared<std::atomic<bool>>();
  return {
      Context::current().derive(FlagKey, Flag),
      [Flag] { *Flag = true; },
  };
}

bool isCancelled(const Context &Ctx) {
  if (auto *Flag = Ctx.get(FlagKey))
    return **Flag;
  return false; // Not in scope of a task.
}

} // namespace clangd
} // namespace clang
