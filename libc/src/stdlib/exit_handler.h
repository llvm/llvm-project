//===-- Implementation header for exit_handler ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_EXIT_HANDLER_H
#define LLVM_LIBC_SRC_STDLIB_EXIT_HANDLER_H

#include "src/__support/CPP/mutex.h" // lock_guard
#include "src/__support/blockstore.h"
#include "src/__support/common.h"
#include "src/__support/fixedvector.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"

namespace LIBC_NAMESPACE_DECL {

using AtExitCallback = void(void *);
using StdCAtExitCallback = void(void);
constexpr size_t CALLBACK_LIST_SIZE_FOR_TESTS = 1024;

struct AtExitUnit {
  AtExitCallback *callback = nullptr;
  void *payload = nullptr;
  void *dso = nullptr;
  LIBC_INLINE constexpr AtExitUnit() = default;
  LIBC_INLINE constexpr AtExitUnit(AtExitCallback *c, void *p,
                                   void *d = nullptr)
      : callback(c), payload(p), dso(d) {}
};

#if defined(LIBC_TARGET_ARCH_IS_GPU)
using ExitCallbackList = FixedVector<AtExitUnit, 64>;
#elif defined(LIBC_COPT_PUBLIC_PACKAGING)
using ExitCallbackList = ReverseOrderBlockStore<AtExitUnit, 32>;
#else
using ExitCallbackList = FixedVector<AtExitUnit, CALLBACK_LIST_SIZE_FOR_TESTS>;
#endif

// This is handled by the 'atexit' implementation and shared by 'at_quick_exit'.
extern Mutex handler_list_mtx;

LIBC_INLINE void stdc_at_exit_func(void *payload) {
  reinterpret_cast<StdCAtExitCallback *>(payload)();
}

LIBC_INLINE AtExitUnit *find_dso_callback(ExitCallbackList &callbacks,
                                          void *dso) {
#if defined(LIBC_COPT_PUBLIC_PACKAGING) && !defined(LIBC_TARGET_ARCH_IS_GPU)
  for (AtExitUnit &unit : callbacks) {
    if (unit.callback && unit.dso == dso)
      return &unit;
  }
#else
  for (auto it = callbacks.rbegin(); it != callbacks.rend(); ++it) {
    if (it->callback && it->dso == dso)
      return &(*it);
  }
#endif
  return nullptr;
}

LIBC_INLINE void call_exit_callbacks(ExitCallbackList &callbacks,
                                     void *dso = nullptr) {
  handler_list_mtx.lock();
  if (!dso) {
    while (!callbacks.empty()) {
      AtExitUnit unit = callbacks.back();
      callbacks.pop_back();
      if (!unit.callback)
        continue;
      handler_list_mtx.unlock();
      unit.callback(unit.payload);
      handler_list_mtx.lock();
    }
    ExitCallbackList::destroy(&callbacks);
    return;
  }

  while (AtExitUnit *unit = find_dso_callback(callbacks, dso)) {
    AtExitCallback *callback = unit->callback;
    void *payload = unit->payload;
    unit->callback = nullptr;
    handler_list_mtx.unlock();
    callback(payload);
    handler_list_mtx.lock();
  }
  handler_list_mtx.unlock();
}

LIBC_INLINE int add_atexit_unit(ExitCallbackList &callbacks,
                                const AtExitUnit &unit) {
  cpp::lock_guard lock(handler_list_mtx);
  if (callbacks.push_back(unit))
    return 0;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_EXIT_HANDLER_H
