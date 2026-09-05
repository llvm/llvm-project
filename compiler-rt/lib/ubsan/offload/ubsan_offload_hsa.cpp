//===-- ubsan_offload_hsa.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ubsan_offload_hsa.h"

#include <dlfcn.h>

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "ubsan_offload.h"
#include "ubsan_offload_rpc.h"
#include "ubsan_offload_symbolize.h"

using namespace __sanitizer;

namespace __ubsan {
namespace {

// Generic helpers to iterate HSA agents and pools.
template <typename ElemTy, typename IterFuncTy, typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void *Data) -> hsa_status_t {
    return (*static_cast<CallbackTy *>(Data))(Elem);
  };
  return Func(L, &Cb);
}

template <typename ElemTy, typename IterFuncTy, typename ArgTy,
          typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, ArgTy Arg, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void *Data) -> hsa_status_t {
    return (*static_cast<CallbackTy *>(Data))(Elem);
  };
  return Func(Arg, L, &Cb);
}

template <typename Elem1Ty, typename Elem2Ty, typename IterFuncTy,
          typename ArgTy, typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, ArgTy Arg, CallbackTy Cb) {
  auto L = [](Elem1Ty A, Elem2Ty B, void *Data) -> hsa_status_t {
    return (*static_cast<CallbackTy *>(Data))(A, B);
  };
  return Func(Arg, L, &Cb);
}

void CheckHsa(hsa_status_t S) {
  if (S == HSA_STATUS_SUCCESS)
    return;
  Report("ERROR: %s: HSA query failed\n", SanitizerToolName);
  Die();
}

} // namespace

template <typename Cb>
void Hsa::ForEachAgentObject(hsa_executable_t Exec, Cb F) {
  Iterate<hsa_executable_t, hsa_loaded_code_object_t>(
      Loader.IterateLoadedCodeObjects, Exec,
      [&](hsa_executable_t, hsa_loaded_code_object_t Obj) {
        u32 Kind = 0;
        if (Loader.GetCodeObjectInfo(
                Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND, &Kind) !=
                HSA_STATUS_SUCCESS ||
            Kind != HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT)
          return HSA_STATUS_SUCCESS;
        hsa_agent_t Agent{};
        if (Loader.GetCodeObjectInfo(
                Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
                &Agent) != HSA_STATUS_SUCCESS ||
            !Agent.handle)
          return HSA_STATUS_SUCCESS;
        F(Obj, Agent);
        return HSA_STATUS_SUCCESS;
      });
}

// Resolve the necessary functions to the real HSA library calls.
bool Hsa::Resolve() {
#define UBSAN_HSA_RESOLVE(Name)                                                \
  Api.Name = reinterpret_cast<decltype(&::Name)>(dlsym(RTLD_NEXT, #Name));     \
  if (!Api.Name)                                                               \
    return false;
  UBSAN_HSA_FUNCTIONS(UBSAN_HSA_RESOLVE)
#undef UBSAN_HSA_RESOLVE
  return true;
}

// Iterate the topology to discover agents. If this fails the interceptors are
// disabled.
bool Hsa::Discover() {
  Agents.clear();
  Devices.clear();
  FineGrainedPool = {};
  hsa_agent_t Cpu{};

  CheckHsa(Iterate<hsa_agent_t>(Api.hsa_iterate_agents, [&](hsa_agent_t Agent) {
    hsa_device_type_t Type;
    if (hsa_status_t S =
            Api.hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type))
      return S;
    Agents.push_back(Agent);
    if (Type == HSA_DEVICE_TYPE_CPU && !Cpu.handle)
      Cpu = Agent;
    if (Type == HSA_DEVICE_TYPE_GPU)
      Devices.push_back(Agent);
    return HSA_STATUS_SUCCESS;
  }));
  if (Devices.empty())
    return false;
  if (!Cpu.handle) {
    Report("ERROR: %s: no CPU HSA agent\n", SanitizerToolName);
    Die();
  }

  CheckHsa(Iterate<hsa_amd_memory_pool_t>(
      Api.hsa_amd_agent_iterate_memory_pools, Cpu,
      [&](hsa_amd_memory_pool_t Pool) {
        hsa_amd_segment_t Seg;
        u32 Flags = 0;
        if (hsa_status_t S = Api.hsa_amd_memory_pool_get_info(
                Pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &Seg))
          return S;
        if (Seg != HSA_AMD_SEGMENT_GLOBAL)
          return HSA_STATUS_SUCCESS;
        if (hsa_status_t S = Api.hsa_amd_memory_pool_get_info(
                Pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &Flags))
          return S;
        if (Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
          FineGrainedPool = Pool;
        return HSA_STATUS_SUCCESS;
      }));
  if (!FineGrainedPool.handle) {
    Report("ERROR: %s: no fine-grained host pool\n", SanitizerToolName);
    Die();
  }
  return true;
}

// Initialize the HSA loader extension used to manage host addresses.
bool Hsa::BindLoader() {
  Loader = {};
  CheckHsa(Api.hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1,
                                                    sizeof(Loader), &Loader));
  return Loader.QueryHostAddress && Loader.IterateLoadedCodeObjects &&
         Loader.GetCodeObjectInfo;
}

bool Hsa::BindDoorbell() {
  if (Api.hsa_amd_signal_create(0, 0, nullptr, 0, &Doorbell) !=
      HSA_STATUS_SUCCESS)
    return false;
  // Mirror of the ROCr amd_signal_t, we extract the KFD interrupt slot to wake
  // the RPC server thread.
  struct AMDSignal {
    int64_t Kind;
    int64_t Value;
    uint64_t EventMailboxPtr;
    uint32_t EventId;
  };
  auto *S = reinterpret_cast<AMDSignal *>(Doorbell.handle);
  DoorbellValue = reinterpret_cast<u64 *>(&S->Value);
  DoorbellMailbox = reinterpret_cast<u64 *>(S->EventMailboxPtr);
  DoorbellEvent = S->EventId;
  return true;
}

bool Hsa::ExecutableInfo(hsa_loaded_code_object_t Obj,
                         hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                         u64 *Out) {
  *Out = 0;
  return Loader.GetCodeObjectInfo(Obj, Attr, Out) == HSA_STATUS_SUCCESS;
}

bool Hsa::AddRef() { return Refs++ == 0; }

bool Hsa::DropRef() {
  if (!Refs)
    return false;
  if (--Refs)
    return false;
  atomic_store(&Active, 0, memory_order_release);
  return true;
}

// Try to initialize all the necessary HSA state to run the sanitizer.
bool Hsa::Init() {
  if (atomic_load(&Active, memory_order_acquire))
    return true;
  if (!Resolve()) {
    Report("ERROR: %s: cannot resolve HSA\n", SanitizerToolName);
    Die();
  }
  if (!Discover())
    return false;
  if (!BindLoader())
    return false;
  if (!BindDoorbell())
    return false;
  VReport(1, "%s: device reporting on %zu GPU(s)\n", SanitizerToolName,
          Devices.size());
  atomic_store(&Active, 1, memory_order_release);
  return true;
}

// Clear the tracked state on HSA shut down.
void Hsa::Shutdown() {
  atomic_store(&Active, 0, memory_order_release);
  StopRpc();
  Lock L(&UbsanOffloadMutex);
  if (Refs)
    return;
  ForgetDeviceImages();
  Executables.clear();
  if (Doorbell.handle) {
    Api.hsa_signal_destroy(Doorbell);
    Doorbell = {};
    DoorbellValue = nullptr;
    DoorbellMailbox = nullptr;
    DoorbellEvent = 0;
  }
  Loader = {};
  Agents.clear();
  Devices.clear();
  FineGrainedPool = {};
}

// Record every executable we come accross for symbolization and address lookup.
void Hsa::RecordExecutable(hsa_executable_t Exec) {
  for (uptr I = 0; I < Executables.size(); ++I)
    if (Executables[I].handle == Exec.handle)
      return;
  Executables.push_back(Exec);
  ForEachAgentObject(Exec, [&](hsa_loaded_code_object_t Obj, hsa_agent_t) {
    u64 LoadBase = 0, LoadSize = 0;
    if (!ExecutableInfo(Obj,
                        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                        &LoadBase) ||
        !ExecutableInfo(Obj,
                        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                        &LoadSize) ||
        !LoadBase || !LoadSize)
      return;

    u64 StorageType = 0, StorageBase = 0, StorageSize = 0;
    const void *Storage = nullptr;
    if (ExecutableInfo(
            Obj,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
            &StorageType) &&
        StorageType == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY &&
        ExecutableInfo(
            Obj,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
            &StorageBase) &&
        ExecutableInfo(
            Obj,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
            &StorageSize) &&
        StorageBase && StorageSize)
      Storage = reinterpret_cast<const void *>(StorageBase);

    TrackDeviceImage((uptr)LoadBase, (uptr)LoadSize, Storage,
                     (uptr)StorageSize);
  });
}

// Remove an executable on runtime unload.
void Hsa::ForgetExecutable(hsa_executable_t Exec) {
  ForEachAgentObject(Exec, [&](hsa_loaded_code_object_t Obj, hsa_agent_t) {
    u64 LoadBase = 0;
    if (ExecutableInfo(Obj,
                       HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                       &LoadBase) &&
        LoadBase)
      ForgetDeviceImage((uptr)LoadBase);
  });
  for (uptr I = 0; I < Executables.size(); ++I) {
    if (Executables[I].handle != Exec.handle)
      continue;
    if (I + 1 != Executables.size())
      Executables[I] = Executables.back();
    Executables.pop_back();
    break;
  }
}

// Allocate coherent 'fine-grained' memory for host and device communication.
bool Hsa::AllocFineGrained(uptr Bytes, void **Out) {
  void *P = nullptr;
  if (!FineGrainedPool.handle ||
      Api.hsa_amd_memory_pool_allocate(FineGrainedPool, Bytes, 0, &P) !=
          HSA_STATUS_SUCCESS ||
      !P)
    return false;
  if (Api.hsa_amd_agents_allow_access(Agents.size(), Agents.data(), nullptr,
                                      P) != HSA_STATUS_SUCCESS) {
    Api.hsa_amd_memory_pool_free(P);
    return false;
  }
  *Out = P;
  return true;
}

void Hsa::Free(void *P) { Api.hsa_amd_memory_pool_free(P); }

bool Hsa::Copy(void *Dst, const void *Src, uptr N) {
  return Api.hsa_memory_copy(Dst, Src, N) == HSA_STATUS_SUCCESS;
}

// The runtime knows the original host address of a device pointer that is
// located inside one of the loaded segments, accesses read-only data we need.
const void *Hsa::HostAddr(uptr Dev) {
  if (!Dev)
    return nullptr;
  const void *Host = nullptr;
  if (Loader.QueryHostAddress(reinterpret_cast<const void *>(Dev), &Host) !=
      HSA_STATUS_SUCCESS)
    return nullptr;
  return Host;
}

// Look up the device address of a named variables in a loaded executable.
bool Hsa::SymbolAddr(hsa_executable_t Exec, const char *Name, hsa_agent_t Agent,
                     u64 *Addr) {
  hsa_executable_symbol_t Symbol;
  if (Api.hsa_executable_get_symbol_by_name(Exec, Name, &Agent, &Symbol) !=
      HSA_STATUS_SUCCESS)
    return false;
  *Addr = 0;
  return Api.hsa_executable_symbol_get_info(
             Symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, Addr) ==
             HSA_STATUS_SUCCESS &&
         *Addr;
}

// The RPC interface is deliberately sized to the hardware parallel limits of
// the device to make deadlock impossible.
void Hsa::RpcInfo(hsa_agent_t Agent, u32 *Lanes, u32 *Waves) {
  u32 CUs = 0, WavesPerCU = 0;
  CheckHsa(Api.hsa_agent_get_info(Agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, Lanes));
  CheckHsa(Api.hsa_agent_get_info(
      Agent,
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
      &CUs));
  CheckHsa(Api.hsa_agent_get_info(
      Agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
      &WavesPerCU));
  *Waves = CUs * WavesPerCU;
}

void Hsa::WaitDoorbell() {
  Api.hsa_signal_wait_scacquire(Doorbell, HSA_SIGNAL_CONDITION_NE, 0,
                                UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
}

void Hsa::KickDoorbell() { Api.hsa_signal_store_screlease(Doorbell, 1); }

Hsa &GetHsa() {
  static Hsa H;
  return H;
}

} // namespace __ubsan
