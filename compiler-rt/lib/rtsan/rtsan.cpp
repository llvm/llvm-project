//===--- rtsan.cpp - Realtime Sanitizer -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "rtsan/rtsan.h"
#include "rtsan/rtsan_assertions.h"
#include "rtsan/rtsan_diagnostics.h"
#include "rtsan/rtsan_flags.h"
#include "rtsan/rtsan_interceptors.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __rtsan;
using namespace __sanitizer;

namespace {
enum class InitializationState : u8 {
  Uninitialized,
  Initializing,
  Initialized,
};
} // namespace

static StaticSpinMutex rtsan_inited_mutex;
static atomic_uint8_t rtsan_initialized = {
    static_cast<u8>(InitializationState::Uninitialized)};

static void SetInitializationState(InitializationState state) {
  atomic_store(&rtsan_initialized, static_cast<u8>(state),
               memory_order_release);
}

static InitializationState GetInitializationState() {
  return static_cast<InitializationState>(
      atomic_load(&rtsan_initialized, memory_order_acquire));
}

static auto OnViolationAction(DiagnosticsInfo info) {
  return [info]() {
    __rtsan::PrintDiagnostics(info);
    if (flags().halt_on_error)
      Die();
  };
}

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_init() {
  CHECK(GetInitializationState() == InitializationState::Uninitialized);
  SetInitializationState(InitializationState::Initializing);

  SanitizerToolName = "RealtimeSanitizer";
  InitializeFlags();
  InitializeInterceptors();

  SetInitializationState(InitializationState::Initialized);
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
  return GetInitializationState() == InitializationState::Initialized;
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
  // While initializing, we need all intercepted functions to behave normally
  if (GetInitializationState() == InitializationState::Initializing)
    return;

  __rtsan_ensure_initialized();
  GET_CALLER_PC_BP;
  ExpectNotRealtime(GetContextForThisThread(),
                    OnViolationAction({DiagnosticsInfoType::InterceptedCall,
                                       func_name, pc, bp}));
}

SANITIZER_INTERFACE_ATTRIBUTE void
__rtsan_notify_blocking_call(const char *func_name) {
  __rtsan_ensure_initialized();
  GET_CALLER_PC_BP;
  ExpectNotRealtime(GetContextForThisThread(),
                    OnViolationAction({DiagnosticsInfoType::BlockingCall,
                                       func_name, pc, bp}));
}

} // extern "C"
