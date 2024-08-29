//===--- rtsan.cpp - Realtime Sanitizer -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <rtsan/rtsan.h>
#include <rtsan/rtsan_context.h>
#include <rtsan/rtsan_interceptors.h>

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_mutex.h"

using namespace __rtsan;
using namespace __sanitizer;

static StaticSpinMutex rtsan_inited_mutex;
static atomic_uint8_t rtsan_initialized = {0};

static void SetInitialized() {
  atomic_store(&rtsan_initialized, 1, memory_order_release);
}

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_init() {
  CHECK(!__rtsan_is_initialized());
  InitializeInterceptors();
  SetInitialized();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_ensure_initialized() {
  if (LIKELY(__rtsan_is_initialized()))
    return;

  SpinMutexLock lock(&rtsan_inited_mutex);

  // Someone may have initialized us while we were waiting for the lock
  if (__rtsan_is_initialized())
    return;

  __rtsan_init();
}

SANITIZER_INTERFACE_ATTRIBUTE bool __rtsan_is_initialized() {
  return atomic_load(&rtsan_initialized, memory_order_acquire) == 1;
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_enter() {
  __rtsan::GetContextForThisThread().RealtimePush();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_exit() {
  __rtsan::GetContextForThisThread().RealtimePop();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_off() {
  __rtsan::GetContextForThisThread().BypassPush();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_on() {
  __rtsan::GetContextForThisThread().BypassPop();
}

SANITIZER_INTERFACE_ATTRIBUTE void
__rtsan_expect_not_realtime(const char *intercepted_function_name) {
  __rtsan_ensure_initialized();
  __rtsan::GetContextForThisThread().ExpectNotRealtime(
      intercepted_function_name);
}

} // extern "C"
