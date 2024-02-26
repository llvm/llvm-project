//===-- GPU implementation of atexit --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atexit.h"
#include "src/__support/common.h"
#include "src/__support/fixedstack.h"

namespace LIBC_NAMESPACE {

namespace {

using AtExitCallback = void(void *);
using StdCAtExitCallback = void(void);

struct AtExitUnit {
  AtExitCallback *callback = nullptr;
  void *payload = nullptr;
  constexpr AtExitUnit() = default;
  constexpr AtExitUnit(AtExitCallback *c, void *p) : callback(c), payload(p) {}
};

// The GPU interface cannot use the standard implementation because it does not
// support the Mutex type. Instead we use a lock free stack with a sufficiently
// large size.
constinit FixedStack<AtExitUnit, CALLBACK_LIST_SIZE_FOR_TESTS> exit_callbacks;

void stdc_at_exit_func(void *payload) {
  reinterpret_cast<StdCAtExitCallback *>(payload)();
}

} // namespace

namespace internal {

void call_exit_callbacks() {
  AtExitUnit unit;
  while (exit_callbacks.pop(unit))
    unit.callback(unit.payload);
}

} // namespace internal

static int add_atexit_unit(const AtExitUnit &unit) {
  if (!exit_callbacks.push(unit))
    return -1;
  return 0;
}

extern "C" int __cxa_atexit(AtExitCallback *callback, void *payload, void *) {
  return add_atexit_unit({callback, payload});
}

LLVM_LIBC_FUNCTION(int, atexit, (StdCAtExitCallback * callback)) {
  return add_atexit_unit(
      {&stdc_at_exit_func, reinterpret_cast<void *>(callback)});
}

} // namespace LIBC_NAMESPACE
