//===--- Cancellation.cpp -----------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cancellation.h"
#include <atomic>

namespace clang {
namespace clangd {

char CancelledError::ID = 0;

// We don't want a cancelable scope to "shadow" an enclosing one.
struct CancelState {
  std::shared_ptr<std::atomic<bool>> Cancelled;
  const CancelState *Parent;
};
static Key<CancelState> StateKey;

std::pair<Context, Canceler> cancelableTask() {
  CancelState State;
  State.Cancelled = std::make_shared<std::atomic<bool>>();
  State.Parent = Context::current().get(StateKey);
  return {
      Context::current().derive(StateKey, State),
      [Flag(State.Cancelled)] { *Flag = true; },
  };
}

bool isCancelled(const Context &Ctx) {
  for (const CancelState *State = Ctx.get(StateKey); State != nullptr;
       State = State->Parent)
    if (State->Cancelled->load())
      return true;
  return false;
}

} // namespace clangd
} // namespace clang
