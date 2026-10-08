//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_offload.h"

#include <dlfcn.h>

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_offload_rpc.h"

namespace __sanitizer {
namespace {

// Generic helpers to iterate HSA agents and pools.
template <typename ElemTy, typename IterFuncTy, typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void* Data) -> hsa_status_t {
    return (*static_cast<CallbackTy*>(Data))(Elem);
  };
  return Func(L, &Cb);
}

template <typename ElemTy, typename IterFuncTy, typename ArgTy,
          typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, ArgTy Arg, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void* Data) -> hsa_status_t {
    return (*static_cast<CallbackTy*>(Data))(Elem);
  };
  return Func(Arg, L, &Cb);
}

template <typename Elem1Ty, typename Elem2Ty, typename IterFuncTy,
          typename ArgTy, typename CallbackTy>
hsa_status_t Iterate(IterFuncTy Func, ArgTy Arg, CallbackTy Cb) {
  auto L = [](Elem1Ty A, Elem2Ty B, void* Data) -> hsa_status_t {
    return (*static_cast<CallbackTy*>(Data))(A, B);
  };
  return Func(Arg, L, &Cb);
}

void CheckHsa(hsa_status_t S) {
  if (S == HSA_STATUS_SUCCESS)
    return;
  Report("ERROR: %s: HSA query failed\n", SanitizerToolName);
  Die();
}

}  // namespace

Offload Offload::Ctx;

Offload& Offload::Get() { return Ctx; }

template <typename Cb>
void Offload::ForEachAgentObject(hsa_executable_t Exec, Cb F) {
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

// Fetch all the real HSA library calls we need.
bool Offload::Resolve() {
#define SANITIZER_HSA_RESOLVE(Name)                                        \
  Api.Name = reinterpret_cast<decltype(&::Name)>(dlsym(RTLD_NEXT, #Name)); \
  if (!Api.Name)                                                           \
    return false;
  SANITIZER_HSA_FUNCTIONS(SANITIZER_HSA_RESOLVE)
#undef SANITIZER_HSA_RESOLVE
  return true;
}

// Iterate the topology to discover agents. If this fails the interceptors are
// disabled.
bool Offload::Discover() {
  DeviceList.clear();
  HostDevice = {};

  auto Info = [&](hsa_agent_t Agent, u32 Attr, void* Out) {
    return Api.hsa_agent_get_info(Agent, static_cast<hsa_agent_info_t>(Attr),
                                  Out) == HSA_STATUS_SUCCESS;
  };
  auto Pool = [&](hsa_agent_t Agent) {
    hsa_amd_memory_pool_t Fine{};
    Iterate<hsa_amd_memory_pool_t>(
        Api.hsa_amd_agent_iterate_memory_pools, Agent,
        [&](hsa_amd_memory_pool_t Mem) {
          hsa_amd_segment_t Seg;
          u32 Flags = 0;
          if (Api.hsa_amd_memory_pool_get_info(Mem,
                                               HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                               &Seg) != HSA_STATUS_SUCCESS ||
              Seg != HSA_AMD_SEGMENT_GLOBAL)
            return HSA_STATUS_SUCCESS;
          if (Api.hsa_amd_memory_pool_get_info(
                  Mem, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &Flags) !=
              HSA_STATUS_SUCCESS)
            return HSA_STATUS_SUCCESS;
          if (Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
            Fine = Mem;
          return HSA_STATUS_SUCCESS;
        });
    return Fine;
  };

  CheckHsa(Iterate<hsa_agent_t>(Api.hsa_iterate_agents, [&](hsa_agent_t Agent) {
    hsa_device_type_t Type;
    if (hsa_status_t S =
            Api.hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type))
      return S;
    Device D = {};
    D.Agent = Agent;
    D.Pool = Pool(Agent);
    if (Type == HSA_DEVICE_TYPE_CPU && !HostDevice.Agent.handle) {
      HostDevice = D;
    } else if (Type == HSA_DEVICE_TYPE_GPU) {
      u32 CUs = 0, WavesPerCU = 0;
      if (!Info(Agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &D.Lanes) || !D.Lanes ||
          !Info(Agent, HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &CUs) ||
          !Info(Agent, HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU, &WavesPerCU) ||
          !CUs || !WavesPerCU)
        CheckHsa(HSA_STATUS_ERROR);
      // The RPC interface is deliberately sized to the hardware parallel
      // limits of the device to make deadlock impossible.
      D.MaxWaves = CUs * WavesPerCU;
      DeviceList.push_back(D);
    }
    return HSA_STATUS_SUCCESS;
  }));
  return HostDevice.Agent.handle && !DeviceList.empty() &&
         HostDevice.Pool.handle;
}

// Initialize the HSA loader extension used to manage host addresses.
bool Offload::BindLoader() {
  Loader = {};
  CheckHsa(Api.hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1,
                                                    sizeof(Loader), &Loader));
  return Loader.QueryHostAddress && Loader.IterateLoadedCodeObjects &&
         Loader.GetCodeObjectInfo;
}

bool Offload::ExecutableInfo(hsa_loaded_code_object_t Obj,
                             hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                             u64* Out) {
  *Out = 0;
  return Loader.GetCodeObjectInfo(Obj, Attr, Out) == HSA_STATUS_SUCCESS;
}

bool Offload::Init() {
  Lock Life(&LifetimeMtx);
  Lock L(&OffloadMtx);
  if (++Refs > 1)
    return Ready();
  if (!Resolve()) {
    Report("ERROR: %s: cannot resolve HSA\n", SanitizerToolName);
    Die();
  }
  if (!Discover() || !BindLoader()) {
    --Refs;
    return false;
  }
  VReport(1, "%s: device reporting on %zu GPU(s)\n", SanitizerToolName,
          DeviceList.size());
  atomic_store(&Active, 1, memory_order_release);
  return true;
}

bool Offload::Release() {
  if (!Refs || --Refs)
    return false;
  atomic_store(&Active, 0, memory_order_release);
  return true;
}

void Offload::Teardown() {
  UntrackImages();
  Loader = {};
  DeviceList.clear();
  HostDevice = {};
}

void Offload::Shutdown() {
  Lock Life(&LifetimeMtx);
  bool Last;
  {
    Lock L(&OffloadMtx);
    Last = Release();
  }
  if (!Last)
    return;
  OffloadRpc::Stop(*this);
  Lock L(&OffloadMtx);
  Teardown();
}

bool Offload::Ready() const {
  return atomic_load(&Active, memory_order_acquire) != 0;
}

void Offload::RegisterHandler(Handler Fn) { OffloadRpc::RegisterHandler(Fn); }

// Record every executable we come across for symbolization and address lookup.
void Offload::TrackExecutable(hsa_executable_t Exec) {
  {
    Lock L(&OffloadMtx);
    if (!Ready())
      return;
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
      const void* Storage = nullptr;
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
        Storage = reinterpret_cast<const void*>(StorageBase);

      TrackImage((uptr)LoadBase, (uptr)LoadSize, Storage, (uptr)StorageSize);
    });
  }
  OffloadRpc::Start(*this, Exec);
}

void Offload::UntrackExecutable(hsa_executable_t Exec) {
  OffloadRpc::Flush();
  Lock L(&OffloadMtx);
  ForEachAgentObject(Exec, [&](hsa_loaded_code_object_t Obj, hsa_agent_t) {
    u64 LoadBase = 0;
    if (ExecutableInfo(Obj,
                       HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                       &LoadBase) &&
        LoadBase)
      UntrackImage((uptr)LoadBase);
  });
}

// Allocate coherent 'fine-grained' memory for host and device communication.
bool Offload::Alloc(const Device& D, uptr Bytes, void** Out) {
  void* P = nullptr;
  if (!D.Pool.handle ||
      Api.hsa_amd_memory_pool_allocate(D.Pool, Bytes, 0, &P) !=
          HSA_STATUS_SUCCESS ||
      !P)
    return false;
  InternalMmapVector<hsa_agent_t> All;
  All.push_back(HostDevice.Agent);
  for (uptr I = 0; I < DeviceList.size(); ++I)
    All.push_back(DeviceList[I].Agent);
  if (Api.hsa_amd_agents_allow_access(All.size(), All.data(), nullptr, P) !=
      HSA_STATUS_SUCCESS) {
    Api.hsa_amd_memory_pool_free(P);
    return false;
  }
  *Out = P;
  return true;
}

bool Offload::GetMemoryPool(hsa_agent_t Agent, hsa_amd_memory_pool_t* Pool) {
  Lock L(&OffloadMtx);
  if (!Ready() || !Pool)
    return false;
  *Pool = {};
  hsa_status_t Status = Iterate<hsa_amd_memory_pool_t>(
      Api.hsa_amd_agent_iterate_memory_pools, Agent,
      [&](hsa_amd_memory_pool_t Mem) {
        hsa_amd_segment_t Segment;
        u32 Flags = 0;
        bool Allowed = false;
        if (Api.hsa_amd_memory_pool_get_info(Mem,
                                             HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                             &Segment) != HSA_STATUS_SUCCESS ||
            Segment != HSA_AMD_SEGMENT_GLOBAL ||
            Api.hsa_amd_memory_pool_get_info(
                Mem, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &Flags) !=
                HSA_STATUS_SUCCESS ||
            !(Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) ||
            Api.hsa_amd_memory_pool_get_info(
                Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                &Allowed) != HSA_STATUS_SUCCESS ||
            !Allowed)
          return HSA_STATUS_SUCCESS;
        *Pool = Mem;
        return HSA_STATUS_SUCCESS;
      });
  return Status == HSA_STATUS_SUCCESS && Pool->handle;
}

bool Offload::Allocate(hsa_amd_memory_pool_t Pool, uptr Bytes, void** Out) {
  Lock L(&OffloadMtx);
  if (!Ready() || !Pool.handle || !Bytes || !Out)
    return false;
  void* P = nullptr;
  if (Api.hsa_amd_memory_pool_allocate(Pool, Bytes, 0, &P) !=
          HSA_STATUS_SUCCESS ||
      !P)
    return false;
  *Out = P;
  return true;
}

void Offload::Free(void* P) { Api.hsa_amd_memory_pool_free(P); }

bool Offload::Copy(void* Dst, const void* Src, uptr N) {
  return Api.hsa_memory_copy(Dst, Src, N) == HSA_STATUS_SUCCESS;
}

// The runtime knows the original host address of a device pointer that is
// located inside one of the loaded segments, accesses read-only data we need.
const void* Offload::HostPointer(uptr DeviceAddr) {
  Lock L(&OffloadMtx);
  if (!Ready())
    return nullptr;
  if (!DeviceAddr)
    return nullptr;
  const void* HostAddr = nullptr;
  if (Loader.QueryHostAddress(reinterpret_cast<const void*>(DeviceAddr),
                              &HostAddr) != HSA_STATUS_SUCCESS)
    return nullptr;
  return HostAddr;
}

bool Offload::Lookup(hsa_executable_t Exec, const char* Name, hsa_agent_t Agent,
                     u64* Addr) {
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

bool Offload::CreateSignal(hsa_signal_t* Out) {
  return Api.hsa_amd_signal_create(0, 0, nullptr, 0, Out) == HSA_STATUS_SUCCESS;
}

void Offload::DestroySignal(hsa_signal_t Sig) { Api.hsa_signal_destroy(Sig); }

void Offload::WaitSignal(hsa_signal_t Sig) {
  Api.hsa_signal_wait_scacquire(Sig, HSA_SIGNAL_CONDITION_NE, 0, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED);
}

void Offload::StoreSignal(hsa_signal_t Sig) {
  Api.hsa_signal_store_screlease(Sig, 1);
}

}  // namespace __sanitizer
