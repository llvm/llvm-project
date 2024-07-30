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

using namespace __rtsan;
using namespace __sanitizer;

static atomic_uint8_t rtsan_initialized{0};
static atomic_uint8_t rtsan_init_is_running{0};

static void SetInitIsRunning(bool is_running) {
  atomic_store(&rtsan_init_is_running, is_running, memory_order_release);
}

static bool IsInitRunning() {
  return atomic_load(&rtsan_init_is_running, memory_order_acquire) == 1;
}

static void SetInitialized() {
  atomic_store(&rtsan_initialized, 1, memory_order_release);
}

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_init() {
  CHECK(!IsInitRunning());
  if (__rtsan_is_initialized())
    return;

  SetInitIsRunning(true);

  InitializeInterceptors();

  SetInitIsRunning(false);
  SetInitialized();
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

} // extern "C"
