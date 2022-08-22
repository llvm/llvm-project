//===-- Implementation of atexit ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atexit.h"
#include "src/__support/blockstore.h"
#include "src/__support/common.h"
#include "src/__support/fixedvector.h"
#include "src/__support/threads/mutex.h"

namespace __llvm_libc {

namespace {

Mutex handler_list_mtx(false, false, false);

using AtExitCallback = void(void *);
using StdCAtExitCallback = void(void);

struct AtExitUnit {
  AtExitCallback *callback = nullptr;
  void *payload = nullptr;
  constexpr AtExitUnit() = default;
  constexpr AtExitUnit(AtExitCallback *c, void *p) : callback(c), payload(p) {}
};

#ifdef LLVM_LIBC_PUBLIC_PACKAGING
using ExitCallbackList = cpp::ReverseOrderBlockStore<AtExitUnit, 32>;
#else
// BlockStore uses dynamic memory allocation. To avoid dynamic memory
// allocation in tests, we use a fixed size callback list when built for
// tests.
// If we use BlockStore, then we will have to pull in malloc etc into
// the tests. While this is not bad, the problem we have currently is
// that LLVM libc' allocator is SCUDO. So, we will end up pulling SCUDO's
// deps also (some of which are not yet available in LLVM libc) into the
// integration tests.
using ExitCallbackList = FixedVector<AtExitUnit, CALLBACK_LIST_SIZE_FOR_TESTS>;
#endif // LLVM_LIBC_PUBLIC_PACKAGING

constinit ExitCallbackList exit_callbacks;

void stdc_at_exit_func(void *payload) {
  reinterpret_cast<StdCAtExitCallback *>(payload)();
}

} // namespace

namespace internal {

void call_exit_callbacks() {
  handler_list_mtx.lock();
  while (!exit_callbacks.empty()) {
    auto unit = exit_callbacks.back();
    exit_callbacks.pop_back();
    handler_list_mtx.unlock();
    unit.callback(unit.payload);
    handler_list_mtx.lock();
  }
  ExitCallbackList::destroy(&exit_callbacks);
}

} // namespace internal

static int add_atexit_unit(const AtExitUnit &unit) {
  // TODO: Use the return value of push_back and bubble it to the public
  // function as error return value. Note that a BlockStore push_back can
  // fail because of allocation failure. Likewise, a FixedVector push_back
  // can fail when it is full.
  handler_list_mtx.lock();
  exit_callbacks.push_back(unit);
  handler_list_mtx.unlock();
  return 0;
}

// TODO: Handle the last dso handle argument.
extern "C" int __cxa_atexit(AtExitCallback *callback, void *payload, void *) {
  return add_atexit_unit({callback, payload});
}

LLVM_LIBC_FUNCTION(int, atexit, (StdCAtExitCallback * callback)) {
  return add_atexit_unit(
      {&stdc_at_exit_func, reinterpret_cast<void *>(callback)});
}

} // namespace __llvm_libc
