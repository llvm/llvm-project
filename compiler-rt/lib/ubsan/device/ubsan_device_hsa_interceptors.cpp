//===-- ubsan_device_hsa_interceptors.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "ubsan_device.h"
#include "ubsan_device_hsa.h"
#include "ubsan_device_rpc.h"
#include "ubsan_device_symbolize.h"
#include "ubsan_diag.h"

#if !SANITIZER_LINUX
#error "Device UBSan reporting is supported on Linux only"
#endif

#if SANITIZER_GLIBC
#pragma weak dlvsym
#endif

using namespace __sanitizer;
using namespace __ubsan;

namespace __ubsan {

Mutex UbsanDeviceMutex;

static StaticSpinMutex InitMutex;
static atomic_uint8_t Initialized;

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;
  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;
  SanitizerToolName = "UndefinedBehaviorSanitizer";
  __ubsan_set_device_symbolize(SymbolizeDevicePc);
  Atexit(ForgetDeviceImages);
  AddDieCallback(ForgetDeviceImages);
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
  if (UNLIKELY(!GetHsa().Ready()))                                             \
    return REAL(name)(__VA_ARGS__);

#define UBSAN_HSA_WRAPS(X)                                                     \
  X(hsa_init)                                                                  \
  X(hsa_shut_down)                                                             \
  X(hsa_executable_freeze)                                                     \
  X(hsa_executable_destroy)

static void *WrapperFor(const char *Name) {
#define UBSAN_HSA_WRAP(Fn)                                                     \
  if (!internal_strcmp(Name, #Fn))                                             \
    return reinterpret_cast<void *>(Fn);
  UBSAN_HSA_WRAPS(UBSAN_HSA_WRAP)
#undef UBSAN_HSA_WRAP
  return nullptr;
}

static bool FromHsa(void *P) {
  Dl_info Info = {};
  if (!dladdr(P, &Info) || !Info.dli_fname)
    return false;
  return internal_strstr(Info.dli_fname, UBSAN_HSA_LIBRARY);
}

static void BindRealDlsym();

// OpenMP and sometimes HIP access HSA through 'dlsym' so we need to intercept
// it here if we want to reliably override its definitions.
INTERCEPTOR(void *, dlsym, void *Handle, const char *Name) {
  Initialize();
  BindRealDlsym();

  // This interceptor interferes with the order of 'RTLD_NEXT'. Force a tail
  // call to bypass this process in the stack.
  if (Handle == RTLD_NEXT) [[clang::musttail]]
    return REAL(dlsym)(Handle, Name);

  void *Sym = REAL(dlsym)(Handle, Name);
  if (!Sym || !Name)
    return Sym;

  void *Wrapper = WrapperFor(Name);
  if (!Wrapper || !FromHsa(Sym))
    return Sym;
  return Wrapper;
}

static void BindRealDlsym() {
  if (LIKELY(REAL(dlsym)))
    return;
#if SANITIZER_GLIBC
  // Need to intercept through 'dlvsym' instead on some platforms.
  static const char *kVers[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.2.5",
                                "GLIBC_2.0"};
  if (dlvsym) {
    for (const char *Ver : kVers) {
      if (void *P = dlvsym(RTLD_NEXT, "dlsym", Ver)) {
        REAL(dlsym) = reinterpret_cast<decltype(REAL(dlsym))>(P);
        return;
      }
    }
  }
#endif
  Report("ERROR: %s: cannot bind dlsym\n", SanitizerToolName);
  Die();
}

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  UBSAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  Lock L(&UbsanDeviceMutex);
  if (GetHsa().AddRef())
    GetHsa().Init();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  UBSAN_HSA_ENTER(hsa_shut_down);

  bool Last;
  {
    Lock L(&UbsanDeviceMutex);
    Last = GetHsa().DropRef();
  }
  if (Last)
    GetHsa().Shutdown();
  return REAL(hsa_shut_down)();
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char *Options) {
  UBSAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status == HSA_STATUS_SUCCESS) {
    {
      Lock L(&UbsanDeviceMutex);
      if (GetHsa().Ready())
        GetHsa().RecordExecutable(Executable);
    }
    StartRpc(Executable);
  }
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  UBSAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  FlushRpc();
  {
    Lock L(&UbsanDeviceMutex);
    GetHsa().ForgetExecutable(Executable);
  }
  return REAL(hsa_executable_destroy)(Executable);
}

extern "C" void __ubsan_device_init() { __ubsan::Initialize(); }

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static void (
    *ubsan_device_preinit)(void) = __ubsan_device_init;
#endif

__attribute__((constructor(0))) static void UbsanDeviceDynInit() {
  __ubsan_device_init();
}
