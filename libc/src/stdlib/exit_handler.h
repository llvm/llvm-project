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
#include "src/__support/threads/mutex.h"

namespace LIBC_NAMESPACE {

using AtExitCallback = void(void *);
using StdCAtExitCallback = void(void);
constexpr size_t CALLBACK_LIST_SIZE_FOR_TESTS = 1024;

struct AtExitUnit {
  AtExitCallback *callback = nullptr;
  void *payload = nullptr;
  LIBC_INLINE constexpr AtExitUnit() = default;
  LIBC_INLINE constexpr AtExitUnit(AtExitCallback *c, void *p)
      : callback(c), payload(p) {}
};

#if defined(LIBC_TARGET_ARCH_IS_GPU)
using ExitCallbackList = FixedVector<AtExitUnit, 64>;
#elif defined(LIBC_COPT_PUBLIC_PACKAGING)
using ExitCallbackList = ReverseOrderBlockStore<AtExitUnit, 32>;
#else
using ExitCallbackList = FixedVector<AtExitUnit, CALLBACK_LIST_SIZE_FOR_TESTS>;
#endif

extern ExitCallbackList atexit_callbacks;
extern ExitCallbackList at_quick_exit_callbacks;

extern Mutex handler_list_mtx;

void stdc_at_exit_func(void *payload);

void call_exit_callbacks(ExitCallbackList &callbacks);

int add_atexit_unit(ExitCallbackList &callbacks, const AtExitUnit &unit);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_EXIT_HANDLER_H
