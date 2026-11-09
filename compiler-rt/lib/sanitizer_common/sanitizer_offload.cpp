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

// Where or not hsa_amd_vmem_handle_create supports fine-grained memory.
constexpr u16 kFineGrainVmemMinor = 33;

bool IsDevice(decltype(&hsa_agent_get_info) Query, hsa_agent_t Agent,
              hsa_device_type_t Type) {
  hsa_device_type_t Got;
  return Query(Agent, HSA_AGENT_INFO_DEVICE, &Got) == HSA_STATUS_SUCCESS &&
         Got == Type;
}

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
#define SANITIZER_HSA_RESOLVE_OPTIONAL(Name) \
  Api.Name = reinterpret_cast<decltype(&::Name)>(dlsym(RTLD_NEXT, #Name));
  SANITIZER_HSA_VMEM_FUNCTIONS(SANITIZER_HSA_RESOLVE_OPTIONAL)
#undef SANITIZER_HSA_RESOLVE_OPTIONAL
  return true;
}

// Iterate the topology to discover agents. If this fails the interceptors are
// disabled.
bool Offload::Discover() {
  DeviceList.clear();
  CpuList.clear();
  PoolList.clear();

  auto Info = [&](hsa_agent_t Agent, u32 Attr, void* Out) {
    return Api.hsa_agent_get_info(Agent, static_cast<hsa_agent_info_t>(Attr),
                                  Out) == HSA_STATUS_SUCCESS;
  };
  auto PoolInfo = [&](hsa_amd_memory_pool_t Mem, u32 Attr, void* Out) {
    return Api.hsa_amd_memory_pool_get_info(
               Mem, static_cast<hsa_amd_memory_pool_info_t>(Attr), Out) ==
           HSA_STATUS_SUCCESS;
  };
  auto RecordPools = [&](hsa_agent_t Agent) {
    Iterate<hsa_amd_memory_pool_t>(
        Api.hsa_amd_agent_iterate_memory_pools, Agent,
        [&](hsa_amd_memory_pool_t Mem) {
          hsa_amd_segment_t Seg;
          u32 Flags = 0;
          bool Allowed = false;
          if (!PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &Seg) ||
              Seg != HSA_AMD_SEGMENT_GLOBAL ||
              !PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &Flags) ||
              !PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                        &Allowed) ||
              !Allowed)
            return HSA_STATUS_SUCCESS;
          MemoryPool P = {Agent, Mem, Flags, 0, 0, false};
          uptr Granule = 0;
          if (PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                       &Granule))
            P.AllocGranule = P.Granule = Granule;
          if (PoolInfo(Mem, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
                       &Granule) &&
              Granule > P.Granule)
            P.Granule = Granule;
          PoolList.push_back(P);
          return HSA_STATUS_SUCCESS;
        });
  };

  CheckHsa(Iterate<hsa_agent_t>(Api.hsa_iterate_agents, [&](hsa_agent_t Agent) {
    hsa_device_type_t Type;
    if (hsa_status_t S =
            Api.hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type))
      return S;
    RecordPools(Agent);
    if (Type == HSA_DEVICE_TYPE_CPU) {
      CpuList.push_back(Agent);
    } else if (Type == HSA_DEVICE_TYPE_GPU) {
      Device D = {};
      D.Agent = Agent;
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

  for (MemoryPool& P : PoolList)
    for (hsa_agent_t Cpu : CpuList)
      P.HostAccess = P.HostAccess || CanAccess(Cpu, P.Pool);
  bool HostFine = false;
  for (const MemoryPool& P : PoolList)
    HostFine |= IsCpu(P.Agent) && P.HostAccess &&
                (P.Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
  return !CpuList.empty() && !DeviceList.empty() && HostFine;
}

// Initialize the HSA loader extension used to manage host addresses.
bool Offload::BindLoader() {
  Loader = {};
  // The loader's size depends on the supported version, a larger size will be
  // left null by the runtime.
  if (Api.hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1,
                                               sizeof(Loader),
                                               &Loader) != HSA_STATUS_SUCCESS) {
    Loader = {};
    CheckHsa(Api.hsa_system_get_major_extension_table(
        HSA_EXTENSION_AMD_LOADER, 1, offsetof(LoaderApi, CreateFromFile),
        &Loader));
  }
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
  CheckHsa(Api.hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR,
                                   &AmdExtMinor));
  VReport(1, "%s: device reporting on %zu GPU(s), AMD extension 1.%u%s\n",
          SanitizerToolName, DeviceList.size(), AmdExtMinor,
          CanLoadAtAddress() ? ", code objects can be placed" : "");
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
  CpuList.clear();
  PoolList.clear();
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

// Allocate from a pool. Grant every agent access if it is reachable.
bool Offload::Alloc(const MemoryPool& Pool, uptr Bytes, void** Out) {
  void* P = nullptr;
  if (!Pool.Pool.handle ||
      Api.hsa_amd_memory_pool_allocate(Pool.Pool, Bytes, 0, &P) !=
          HSA_STATUS_SUCCESS ||
      !P)
    return false;
  InternalMmapVector<hsa_agent_t> All;
  for (hsa_agent_t Cpu : CpuList)
    if (CanAccess(Cpu, Pool.Pool))
      All.push_back(Cpu);
  for (const Device& D : DeviceList) All.push_back(D.Agent);
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

bool Offload::Copy(void* Dst, const void* Src, uptr N) const {
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

bool Offload::IsGpu(hsa_agent_t Agent) const {
  return IsDevice(Api.hsa_agent_get_info, Agent, HSA_DEVICE_TYPE_GPU);
}

bool Offload::IsCpu(hsa_agent_t Agent) const {
  return IsDevice(Api.hsa_agent_get_info, Agent, HSA_DEVICE_TYPE_CPU);
}

bool Offload::CanAccess(hsa_agent_t Agent, hsa_amd_memory_pool_t Pool) const {
  hsa_amd_memory_pool_access_t Access =
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  return Api.hsa_amd_agent_memory_pool_get_info(
             Agent, Pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &Access) ==
             HSA_STATUS_SUCCESS &&
         Access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
}

// Requires HSA AMD version minor 0.33.
bool Offload::FineGrainVmem() const {
  return AmdExtMinor >= kFineGrainVmemMinor;
}

// Supports loader extension API 1.04.
bool Offload::CanLoadAtAddress() const {
  return Loader.GetLoadSize && Loader.LoadAtAddress;
}

// Requires loader extension API 1.04.
bool Offload::CodeObjectLoadSize(hsa_code_object_reader_t Reader,
                                 uptr* Size) const {
  DCHECK(Loader.GetLoadSize);
  size_t Bytes = 0;
  if (!Loader.GetLoadSize ||
      Loader.GetLoadSize(Reader, &Bytes) != HSA_STATUS_SUCCESS || !Bytes)
    return false;
  *Size = (uptr)Bytes;
  return true;
}

// Requires loader extension API 1.04.
bool Offload::LoadCodeObjectAt(hsa_executable_t Exec, hsa_agent_t Agent,
                               hsa_code_object_reader_t Reader,
                               const char* Options, void* Address, uptr Size,
                               hsa_loaded_code_object_t* Loaded) const {
  DCHECK(Loader.LoadAtAddress);
  return Loader.LoadAtAddress &&
         Loader.LoadAtAddress(Exec, Agent, Reader, Options, Address, Size,
                              Loaded) == HSA_STATUS_SUCCESS;
}

bool Offload::HasVirtualMemory() const {
#define SANITIZER_HSA_PRESENT(Name) &&Api.Name
  return true SANITIZER_HSA_VMEM_FUNCTIONS(SANITIZER_HSA_PRESENT);
#undef SANITIZER_HSA_PRESENT
}

// Reserve exactly [Addr, Addr + Size), or nothing.
bool Offload::ReserveAddress(uptr Addr, uptr Size) {
  // The HSA call here is allowed to return an HSA_STATUS_SUCCESS if it failed
  // to reserve the specified range. The sanitizer requires an exact size, so we
  // need to detect this and return it as a failure..
  auto Exact = [&](u64 Flags, uptr Align) {
    void* Va = nullptr;
    if (!Align || Api.hsa_amd_vmem_address_reserve_align(
                      &Va, Size, Addr, Align, Flags) != HSA_STATUS_SUCCESS)
      return false;
    if (reinterpret_cast<uptr>(Va) == Addr)
      return true;
    Api.hsa_amd_vmem_address_free(Va, Size);
    return false;
  };
  return Exact(HSA_AMD_VMEM_ADDRESS_NO_REGISTER, GetPageSizeCached()) ||
         Exact(0, Addr & -Addr);
}

void Offload::FreeAddress(uptr Addr, uptr Size) {
  Api.hsa_amd_vmem_address_free(reinterpret_cast<void*>(Addr), Size);
}

bool Offload::CreateHandle(hsa_amd_memory_pool_t Pool, uptr Size, u64 Flags,
                           u64* Handle) {
  hsa_amd_vmem_alloc_handle_t H;
  if (Api.hsa_amd_vmem_handle_create(Pool, Size, MEMORY_TYPE_NONE, Flags, &H) !=
      HSA_STATUS_SUCCESS)
    return false;
  *Handle = H.handle;
  return true;
}

void Offload::ReleaseHandle(u64 Handle) {
  Api.hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t{Handle});
}

bool Offload::Map(uptr Addr, uptr Size, uptr Offset, u64 Handle) {
  return Api.hsa_amd_vmem_map(reinterpret_cast<void*>(Addr), Size, Offset,
                              hsa_amd_vmem_alloc_handle_t{Handle},
                              0) == HSA_STATUS_SUCCESS;
}

void Offload::Unmap(uptr Addr, uptr Size) {
  Api.hsa_amd_vmem_unmap(reinterpret_cast<void*>(Addr), Size);
}

bool Offload::SetAccess(uptr Addr, uptr Size,
                        const hsa_amd_memory_access_desc_t* Desc, uptr Count) {
  return Api.hsa_amd_vmem_set_access(reinterpret_cast<void*>(Addr), Size, Desc,
                                     Count) == HSA_STATUS_SUCCESS;
}

// dlsym(RTLD_DEFAULT) follows load order. hsa_init resolving inside libhsa
// means this runtime was loaded too late to interpose.
void WarnIfHsaNotInterposed() {
  void* Sym = dlsym(RTLD_DEFAULT, "hsa_init");
  if (!Sym)
    return;
  Dl_info Info = {};
  if (!dladdr(Sym, &Info) || !Info.dli_fname ||
      !internal_strstr(Info.dli_fname, SANITIZER_HSA_LIBRARY))
    return;
  Report(
      "WARNING: %s: the runtime is loaded too late to intercept HSA. "
      "GPU checking will not work. Link the runtime first or use "
      "LD_PRELOAD.\n",
      SanitizerToolName);
}

}  // namespace __sanitizer
