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
#include <rtsan/rtsan_assertions.h>
#include <rtsan/rtsan_diagnostics.h>
#include <rtsan/rtsan_flags.h>
#include <rtsan/rtsan_interceptors.h>

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __rtsan;
using namespace __sanitizer;

static StaticSpinMutex rtsan_inited_mutex;
static atomic_uint8_t rtsan_initialized = {0};

static void SetInitialized() {
  atomic_store(&rtsan_initialized, 1, memory_order_release);
}

static auto PrintDiagnosticsAndDieAction(DiagnosticsInfo info) {
  return [info]() {
    __rtsan::PrintDiagnostics(info);
    Die();
  };
}

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_init() {
  CHECK(!__rtsan_is_initialized());

  SanitizerToolName = "RealtimeSanitizer";
  InitializeFlags();
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

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_disable() {
  __rtsan::GetContextForThisThread().BypassPush();
}

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_enable() {
  __rtsan::GetContextForThisThread().BypassPop();
}

SANITIZER_INTERFACE_ATTRIBUTE void
__rtsan_notify_intercepted_call(const char *func_name) {
  __rtsan_ensure_initialized();
  GET_CALLER_PC_BP;
  ExpectNotRealtime(
      GetContextForThisThread(),
      PrintDiagnosticsAndDieAction(
          {DiagnosticsInfoType::InterceptedCall, func_name, pc, bp}));
}

SANITIZER_INTERFACE_ATTRIBUTE void
__rtsan_notify_blocking_call(const char *func_name) {
  __rtsan_ensure_initialized();
  GET_CALLER_PC_BP;
  ExpectNotRealtime(
      GetContextForThisThread(),
      PrintDiagnosticsAndDieAction(
          {DiagnosticsInfoType::BlockingCall, func_name, pc, bp}));
}

} // extern "C"
