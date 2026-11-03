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
  atomic_store(&Initialized, 1, memory_order_release);
  Symbolizer::LateInitialize();
}

} // namespace

// DSOs with a static runtime all export these wrappers and interpose onto the
// first, so 'RTLD_NEXT' can loop back into it. Resolve from HSA directly
// without loading it.
static void *HsaSymbol(const char *Name) {
  SpinMutexLock L(&HsaMutex);
  constexpr const char *Libs[] = {SANITIZER_HSA_LIBRARY ".so.1",
                                  SANITIZER_HSA_LIBRARY ".so"};
  for (const char *Lib : Libs)
    if (!HsaHandle)
      HsaHandle = dlopen(Lib, RTLD_LAZY | RTLD_NOLOAD);
  return HsaHandle ? dlsym(HsaHandle, Name) : nullptr;
}

template <typename T> static T HsaFunction(const char *Name) {
  return reinterpret_cast<T>(HsaSymbol(Name));
}

// The shared runtime exports these to every program, act as if HSA is absent
// when it is not loaded.
#define CSAN_HSA_ENTER(name)                                                   \
  Initialize();                                                                \
  if (UNLIKELY(!REAL(name))) {                                                 \
    REAL(name) = HsaFunction<decltype(REAL(name))>(#name);                     \
    if (UNLIKELY(!REAL(name))) {                                               \
      VReport(1, "%s: cannot find %s in this process\n", SanitizerToolName,    \
              #name);                                                          \
      return HSA_STATUS_ERROR;                                                 \
    }                                                                          \
  }

#define CSAN_HSA_FORWARD(name, ...)                                            \
  CSAN_HSA_ENTER(name);                                                        \
  if (UNLIKELY(!Offload::Get().Ready()))                                       \
    return REAL(name)(__VA_ARGS__);

static bool FromHsa(void *P) {
  Dl_info Info = {};
  if (!dladdr(P, &Info) || !Info.dli_fname)
    return false;
  return internal_strstr(Info.dli_fname, SANITIZER_HSA_LIBRARY);
}

// Callers bind to whichever 'hsa_init' comes first, if HSA was loaded before
// the runtime the interceptors are bypassed.
static void CheckInterposed() {
  void *Sym = dlsym(RTLD_DEFAULT, "hsa_init");
  if (!Sym || !FromHsa(Sym))
    return;
  Report("WARNING: %s: the runtime is loaded too late to intercept HSA, GPU "
         "races will not be reported. Link the runtime first or use "
         "LD_PRELOAD.\n",
         SanitizerToolName);
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

__attribute__((constructor(0))) static void CsanOffloadDynInit() {
  __csan_offload_init();
  CheckInterposed();
}
