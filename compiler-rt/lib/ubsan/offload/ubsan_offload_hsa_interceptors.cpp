//===-- ubsan_offload_hsa_interceptors.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "ubsan_diag.h"
#include "ubsan_offload.h"

#if !SANITIZER_LINUX
#error "Offload UBSan reporting is supported on Linux only"
#endif

using namespace __sanitizer;
using namespace __ubsan;

namespace __ubsan {

static StaticSpinMutex InitMutex;
static atomic_uint8_t Initialized;

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;
  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;
  SanitizerToolName = "UndefinedBehaviorSanitizer";
  __ubsan_set_offload_symbolize(
      [](uptr PC) { return Offload::Get().Symbolize(PC); });
  Offload::Get().RegisterHandler(HandleOffloadReport);
  Atexit([] { Offload::Get().UntrackImages(); });
  AddDieCallback([] { Offload::Get().UntrackImages(); });
  atomic_store(&Initialized, 1, memory_order_release);
}

} // namespace __ubsan

#define UBSAN_HSA_ENTER(name)                                                  \
  Initialize();                                                                \
  if (UNLIKELY(!REAL(name))) {                                                 \
    INTERCEPT_FUNCTION(name);                                                  \
    if (UNLIKELY(!REAL(name))) {                                               \
      Report("ERROR: %s: cannot find %s in this process\n", SanitizerToolName, \
             #name);                                                           \
      Die();                                                                   \
    }                                                                          \
  }

#define UBSAN_HSA_FORWARD(name, ...)                                           \
  UBSAN_HSA_ENTER(name);                                                       \
  if (UNLIKELY(!Offload::Get().Ready()))                                       \
    return REAL(name)(__VA_ARGS__);

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  UBSAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  Offload::Get().Init();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  UBSAN_HSA_ENTER(hsa_shut_down);

  Offload::Get().Shutdown();
  return REAL(hsa_shut_down)();
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char *Options) {
  UBSAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status == HSA_STATUS_SUCCESS)
    Offload::Get().TrackExecutable(Executable);
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  UBSAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  Offload::Get().UntrackExecutable(Executable);
  return REAL(hsa_executable_destroy)(Executable);
}

extern "C" void __ubsan_offload_init() { __ubsan::Initialize(); }

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static void (
    *ubsan_offload_preinit)(void) = __ubsan_offload_init;
#endif

__attribute__((constructor(0))) static void UbsanOffloadDynInit() {
  __ubsan_offload_init();
}
