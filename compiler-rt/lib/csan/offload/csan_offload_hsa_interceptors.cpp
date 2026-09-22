//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// HSA interceptors for host-side ConcurrencySanitizer offload support.
///
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

#include "csan_offload.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "sanitizer_common/sanitizer_vector.h"

#if !SANITIZER_LINUX
#error "Offload CSan reporting is supported on Linux only"
#endif

#if SANITIZER_GLIBC
#pragma weak dlvsym
#endif

using namespace __sanitizer;
using namespace __csan;

namespace {

static StaticSpinMutex InitMutex;
static StaticSpinMutex HsaMutex;
static atomic_uint8_t Initialized;
static void *HsaHandle;

struct DeviceAllocation {
  hsa_agent_t Agent;
  void *Ptr;
};

Mutex AllocationMutex;
InternalMmapVectorNoCtor<DeviceAllocation> Allocations;
uptr HsaRefs;

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;
  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;
  SanitizerToolName = "ConcurrencySanitizer";
  CacheBinaryName();
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    if (const char *Path = GetEnv("CSAN_SYMBOLIZER_PATH"))
      cf.external_symbolizer_path = Path;
    cf.stack_trace_format = "    #%n %f %S (%p)";
    OverrideCommonFlags(cf);
  }
  FlagParser Parser;
  RegisterCommonFlags(&Parser);
  Parser.ParseStringFromEnv("CSAN_OPTIONS");
  InitializeCommonFlags();
  Offload::Get().RegisterHandler(HandleOffloadReport);
  Atexit([] { Offload::Get().UntrackImages(); });
  AddDieCallback([] { Offload::Get().UntrackImages(); });

  // Mark ready before LateInitialize, as it can be reentrant through dlsym.
  atomic_store(&Initialized, 1, memory_order_release);
  Symbolizer::LateInitialize();
}

} // namespace

static void BindRealDlsym();
static void *HsaSymbol(const char *Name);

#define CSAN_HSA_ENTER(name)                                                   \
  Initialize();                                                                \
  if (UNLIKELY(!REAL(name) || REAL(name) == name)) {                           \
    REAL(name) = reinterpret_cast<decltype(REAL(name))>(HsaSymbol(#name));     \
    if (UNLIKELY(!REAL(name) || REAL(name) == name)) {                         \
      Report("ERROR: %s: cannot find %s in this process\n", SanitizerToolName, \
             #name);                                                           \
      Die();                                                                   \
    }                                                                          \
  }

#define CSAN_HSA_FORWARD(name, ...)                                            \
  CSAN_HSA_ENTER(name);                                                        \
  if (UNLIKELY(!Offload::Get().Ready()))                                       \
    return REAL(name)(__VA_ARGS__);

// PPC cannot transparently tail-call an indirect dlsym target for RTLD_NEXT.
#if !SANITIZER_PPC
#define CSAN_HSA_WRAPS(X)                                                      \
  X(hsa_init)                                                                  \
  X(hsa_shut_down)                                                             \
  X(hsa_executable_freeze)                                                     \
  X(hsa_executable_destroy)

static void *WrapperFor(const char *Name) {
#define CSAN_HSA_WRAP(Fn)                                                      \
  if (!internal_strcmp(Name, #Fn))                                             \
    return reinterpret_cast<void *>(Fn);
  CSAN_HSA_WRAPS(CSAN_HSA_WRAP)
#undef CSAN_HSA_WRAP
  return nullptr;
}

static bool FromHsa(void *P) {
  Dl_info Info = {};
  if (!dladdr(P, &Info) || !Info.dli_fname)
    return false;
  return internal_strstr(Info.dli_fname, SANITIZER_HSA_LIBRARY);
}

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
#else
DEFINE_REAL(void *, dlsym, void *, const char *)
#endif

static void BindRealDlsym() {
  if (LIKELY(REAL(dlsym)))
    return;
#if SANITIZER_GLIBC
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

static void *HsaSymbol(const char *Name) {
  BindRealDlsym();
  if (!HsaHandle) {
    SpinMutexLock L(&HsaMutex);
    if (HsaHandle)
      return REAL(dlsym)(HsaHandle, Name);
    constexpr const char *Names[] = {"libhsa-runtime64.so.1",
                                     "libhsa-runtime64.so"};
    for (const char *Name : Names)
      if (void *H = dlopen(Name, RTLD_LAZY | RTLD_NOLOAD))
        HsaHandle = H;
    for (const char *Name : Names)
      if (!HsaHandle)
        HsaHandle = dlopen(Name, RTLD_LAZY | RTLD_LOCAL);
  }
  return HsaHandle ? REAL(dlsym)(HsaHandle, Name) : nullptr;
}

template <typename T> static T HsaFunction(const char *Name) {
  return reinterpret_cast<T>(HsaSymbol(Name));
}

static bool Lookup(hsa_executable_t Executable, const char *Name,
                   hsa_agent_t Agent, u64 *Addr) {
  auto GetSymbol = HsaFunction<decltype(&hsa_executable_get_symbol_by_name)>(
      "hsa_executable_get_symbol_by_name");
  auto GetInfo = HsaFunction<decltype(&hsa_executable_symbol_get_info)>(
      "hsa_executable_symbol_get_info");
  if (!GetSymbol || !GetInfo)
    return false;

  hsa_executable_symbol_t Symbol;
  if (GetSymbol(Executable, Name, &Agent, &Symbol) != HSA_STATUS_SUCCESS)
    return false;
  *Addr = 0;
  return GetInfo(Symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, Addr) ==
             HSA_STATUS_SUCCESS &&
         *Addr;
}

static DeviceAllocation *FindAllocation(hsa_agent_t Agent) {
  for (uptr I = 0; I < Allocations.size(); ++I)
    if (Allocations[I].Agent.handle == Agent.handle)
      return &Allocations[I];
  return nullptr;
}

struct BindContext {
  hsa_executable_t Executable;
  bool Found;
  bool Success;
};

// The watchpoint table is shared between all executables loaded on the device.
static hsa_status_t BindAgent(hsa_agent_t Agent, void *Data) {
  BindContext &Ctx = *reinterpret_cast<BindContext *>(Data);
  auto AgentInfo =
      HsaFunction<decltype(&hsa_agent_get_info)>("hsa_agent_get_info");
  auto Copy = HsaFunction<decltype(&hsa_memory_copy)>("hsa_memory_copy");
  auto Free = HsaFunction<decltype(&hsa_amd_memory_pool_free)>(
      "hsa_amd_memory_pool_free");
  if (!AgentInfo || !Copy || !Free) {
    Ctx.Success = false;
    return HSA_STATUS_SUCCESS;
  }

  hsa_device_type_t Type;
  if (AgentInfo(Agent, HSA_AGENT_INFO_DEVICE, &Type) != HSA_STATUS_SUCCESS ||
      Type != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;

  u64 PointerAddr = 0;
  if (!Lookup(Ctx.Executable, "__csan_watchpoint_table", Agent, &PointerAddr))
    return HSA_STATUS_SUCCESS;
  Ctx.Found = true;

  // Create an allocation for the table if one does not already exist.
  DeviceAllocation *Allocation = FindAllocation(Agent);
  if (!Allocation) {
    hsa_amd_memory_pool_t Pool;
    void *Ptr = nullptr;
    if (!Offload::Get().GetMemoryPool(Agent, &Pool) ||
        !Offload::Get().Allocate(Pool, CSAN_WATCHPOINT_TABLE_BYTES, &Ptr)) {
      Ctx.Success = false;
      return HSA_STATUS_SUCCESS;
    }

    // Fresh pages from MMap are always zero initialized, as required.
    void *Zeros =
        MmapOrDie(CSAN_WATCHPOINT_TABLE_BYTES, "CSan watchpoint table");
    bool Copied =
        Copy(Ptr, Zeros, CSAN_WATCHPOINT_TABLE_BYTES) == HSA_STATUS_SUCCESS;
    UnmapOrDie(Zeros, CSAN_WATCHPOINT_TABLE_BYTES);
    if (!Copied) {
      Free(Ptr);
      Ctx.Success = false;
      return HSA_STATUS_SUCCESS;
    }
    Allocations.push_back({Agent, Ptr});
    Allocation = &Allocations.back();
  }

  // Update the pointer on the device to the associated allocation.
  if (Copy(reinterpret_cast<void *>(PointerAddr), &Allocation->Ptr,
           sizeof(Allocation->Ptr)) != HSA_STATUS_SUCCESS)
    Ctx.Success = false;
  return HSA_STATUS_SUCCESS;
}

static void BindWatchpointTable(hsa_executable_t Executable) {
  auto Iterate =
      HsaFunction<decltype(&hsa_iterate_agents)>("hsa_iterate_agents");
  if (!Iterate)
    return;

  // Check if we need to set up the watchpoint table.
  Lock L(&AllocationMutex);
  BindContext Ctx = {Executable, false, true};
  if (Iterate(BindAgent, &Ctx) != HSA_STATUS_SUCCESS)
    Ctx.Success = false;
  if (Ctx.Found && !Ctx.Success) {
    Report("ERROR: %s: cannot initialize device watchpoint table\n",
           SanitizerToolName);
    Die();
  }
}

static void RetainAllocations() {
  Lock L(&AllocationMutex);
  ++HsaRefs;
}

static void ReleaseAllocations() {
  Lock L(&AllocationMutex);
  if (!HsaRefs || --HsaRefs)
    return;
  auto Free = HsaFunction<decltype(&hsa_amd_memory_pool_free)>(
      "hsa_amd_memory_pool_free");
  if (Free)
    for (uptr I = 0; I < Allocations.size(); ++I)
      Free(Allocations[I].Ptr);
  Allocations.clear();
}

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  CSAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  if (!Offload::Get().Init()) {
    Report("ERROR: %s: cannot initialize HSA offload support\n",
           SanitizerToolName);
    Die();
  }
  RetainAllocations();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  CSAN_HSA_ENTER(hsa_shut_down);

  ReleaseAllocations();
  Offload::Get().Shutdown();
  return REAL(hsa_shut_down)();
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char *Options) {
  CSAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status == HSA_STATUS_SUCCESS) {
    BindWatchpointTable(Executable);
    Offload::Get().TrackExecutable(Executable);
  }
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  CSAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  Offload::Get().UntrackExecutable(Executable);
  return REAL(hsa_executable_destroy)(Executable);
}

extern "C" void __csan_offload_init() { Initialize(); }

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static void (
    *csan_offload_preinit)(void) = __csan_offload_init;
#endif

__attribute__((constructor(0))) static void CsanOffloadDynInit() {
  __csan_offload_init();
}
