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

// PM4 type-3 header. Count is the body size minus one. Bit 1 is the compute
// shader type.
u32 Pm4Header(u32 Opcode, u32 Dwords) {
  return (1u << 1) | (Opcode << 8) | ((Dwords - 2) << 16) | (3u << 30);
}

// One WRITE_DATA of a contiguous run. dst_sel = TC_L2, wr_confirm, address
// increments per dword, cache policy LRU. The control word is the same from
// gfx9 onward; a qword is two data dwords, a longer run is more of them.
//
// The type-3 count is 14 bits, so one WRITE_DATA stops at 65520 bytes. The
// indirect-buffer size is 20 bits, so one jump carries 63 of those packets
// and one completion signal. Every packet confirms before the next starts.
constexpr u32 kWritePacketDwords = 16384;
constexpr u32 kPacketsPerIb = 63;
constexpr u32 kIbDwords = kWritePacketDwords * kPacketsPerIb;
constexpr uptr kPm4IbBytes = static_cast<uptr>(kIbDwords) * sizeof(u32);
constexpr uptr kWriteChunkBytes = (kWritePacketDwords - 4) * sizeof(u32);
static_assert(kWritePacketDwords - 2 <= 0x3FFFu, "WRITE_DATA count is 14 bits");
static_assert(kIbDwords <= 0xFFFFFu, "indirect buffer size is 20 bits");
static_assert(kWriteChunkBytes % sizeof(u64) == 0, "entries are 8 bytes");

u32 EmitWriteData(u32* Out, uptr Addr, const void* Src, uptr Bytes) {
  u32 DataDwords = Bytes / sizeof(u32);
  u32 PacketDwords = DataDwords + 4;
  Out[0] = Pm4Header(0x37, PacketDwords);
  Out[1] = (2u << 8) | (1u << 20);
  Out[2] = static_cast<u32>(Addr);
  Out[3] = static_cast<u32>(Addr >> 32);
  if (Src)
    internal_memcpy(Out + 4, Src, Bytes);
  else
    internal_memset(Out + 4, 0, Bytes);
  return PacketDwords;
}

// Vendor AQL packet that jumps to a PM4 indirect buffer. dw_cnt_remain counts
// the dwords after that field; the firmware expects 0xA.
struct AmdAqlPm4Ib {
  u16 Header;
  u16 VenHdr;
  u32 IbJump[4];
  u32 DwCntRemain;
  u32 Reserved[8];
  hsa_signal_t Completion;
};
static_assert(sizeof(AmdAqlPm4Ib) == 64, "vendor PM4 packet");
static_assert(offsetof(AmdAqlPm4Ib, Completion) == 56, "vendor PM4 packet");

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

bool Offload::DescribeCodeObject(hsa_loaded_code_object_t Obj,
                                 hsa_agent_t Agent, CodeObject* Out) {
  *Out = {};
  Out->Agent = Agent;
  u64 LoadBase = 0, LoadSize = 0;
  if (!ExecutableInfo(Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                      &LoadBase) ||
      !ExecutableInfo(Obj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                      &LoadSize) ||
      !LoadBase || !LoadSize)
    return false;
  Out->LoadBase = (uptr)LoadBase;
  Out->LoadSize = (uptr)LoadSize;

  u64 StorageType = 0, StorageBase = 0, StorageSize = 0;
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
      StorageBase && StorageSize) {
    Out->Storage = reinterpret_cast<const void*>(StorageBase);
    Out->StorageSize = (uptr)StorageSize;
  }
  return true;
}

void Offload::VisitCodeObjects(hsa_executable_t Exec,
                               void (*Fn)(const CodeObject&, void*),
                               void* Arg) {
  InternalMmapVector<CodeObject> Objects;
  {
    Lock L(&OffloadMtx);
    if (!Ready())
      return;
    ForEachAgentObject(Exec,
                       [&](hsa_loaded_code_object_t Obj, hsa_agent_t Agent) {
                         CodeObject Desc;
                         if (DescribeCodeObject(Obj, Agent, &Desc))
                           Objects.push_back(Desc);
                       });
  }
  for (const CodeObject& Obj : Objects) Fn(Obj, Arg);
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
  if (!CreateCommands()) {
    Teardown();
    --Refs;
    return false;
  }
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
  for (Device& D : DeviceList) {
    if (D.Queue)
      Api.hsa_queue_destroy(D.Queue);
    if (D.Signal.handle)
      DestroySignal(D.Signal);
  }
  if (Ib)
    Free(Ib);
  Ib = nullptr;
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
    ForEachAgentObject(Exec,
                       [&](hsa_loaded_code_object_t Obj, hsa_agent_t Agent) {
                         CodeObject Desc;
                         if (DescribeCodeObject(Obj, Agent, &Desc))
                           TrackImage(Desc.LoadBase, Desc.LoadSize,
                                      Desc.Storage, Desc.StorageSize);
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
bool Offload::Alloc(const MemoryPool& Pool, uptr Bytes, void** Out, u64 Flags) {
  void* P = nullptr;
  if (!Pool.Pool.handle ||
      Api.hsa_amd_memory_pool_allocate(Pool.Pool, Bytes, Flags, &P) !=
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

bool Offload::PoolForAgent(hsa_agent_t Agent, u32 Match, MemoryPool* Out) {
  if (!Out || !Agent.handle || !Match)
    return false;
  Lock L(&OffloadMtx);
  if (!Ready())
    return false;
  const MemoryPool* Fallback = nullptr;
  for (const MemoryPool& P : PoolList) {
    if (P.Agent.handle != Agent.handle || !(P.Flags & Match))
      continue;
    if (P.Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
      *Out = P;
      return true;
    }
    if (!Fallback)
      Fallback = &P;
  }
  if (!Fallback)
    return false;
  *Out = *Fallback;
  return true;
}

bool Offload::FindPool(hsa_amd_memory_pool_t Pool, MemoryPool* Out) {
  if (!Out || !Pool.handle)
    return false;
  Lock L(&OffloadMtx);
  if (!Ready())
    return false;
  for (const MemoryPool& P : PoolList) {
    if (P.Pool.handle != Pool.handle)
      continue;
    *Out = P;
    return true;
  }
  return false;
}

bool Offload::HostPool(MemoryPool* Out) {
  if (!Out)
    return false;
  Lock L(&OffloadMtx);
  if (!Ready())
    return false;
  for (const MemoryPool& P : PoolList) {
    bool Gpu = false;
    for (const Device& D : DeviceList)
      if (D.Agent.handle == P.Agent.handle)
        Gpu = true;
    if (Gpu || !(P.Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED))
      continue;
    *Out = P;
    return true;
  }
  return false;
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

// One queue per GPU, at the agent's minimum depth, and one executable
// fine-grained buffer. Both stay until shutdown. A store fills the buffer
// and rings the queue.
bool Offload::CreateCommands() {
  const MemoryPool* Fine = nullptr;
  for (const MemoryPool& P : PoolList) {
    if (IsCpu(P.Agent) && P.HostAccess &&
        (P.Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
      Fine = &P;
      break;
    }
  }
  if (!Fine ||
      !Alloc(*Fine, kPm4IbBytes, &Ib, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG)) {
    Report("%s: device write buffer unavailable\n", SanitizerToolName);
    return false;
  }
  for (Device& D : DeviceList) {
    u32 Packets = 0;
    if (Api.hsa_agent_get_info(D.Agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                               &Packets) != HSA_STATUS_SUCCESS ||
        !Packets ||
        Api.hsa_queue_create(D.Agent, Packets, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &D.Queue) != HSA_STATUS_SUCCESS ||
        !CreateSignal(&D.Signal)) {
      if (D.Queue) {
        Api.hsa_queue_destroy(D.Queue);
        D.Queue = nullptr;
      }
      D.Signal = {};
      Report("%s: device write queue unavailable\n", SanitizerToolName);
      return false;
    }
  }
  return true;
}

void Offload::SubmitIb(Device& D, u32 Dwords) {
  AmdAqlPm4Ib Pkt = {};
  Pkt.VenHdr = 1;
  uptr Base = reinterpret_cast<uptr>(Ib);
  // The AQL firmware rejects the compute bit on this jump.
  Pkt.IbJump[0] = Pm4Header(0x3F, 4) & ~(1u << 1);
  Pkt.IbJump[1] = static_cast<u32>(Base);
  Pkt.IbJump[2] = static_cast<u32>(Base >> 32);
  Pkt.IbJump[3] = (Dwords & 0xFFFFFu) | (1u << 23);
  Pkt.DwCntRemain = 0xA;
  Pkt.Completion = D.Signal;

  Api.hsa_signal_store_screlease(D.Signal, 1);
  u64 Index = Api.hsa_queue_add_write_index_screlease(D.Queue, 1);
  auto* Slot = static_cast<u8*>(D.Queue->base_address) +
               (Index % D.Queue->size) * sizeof(Pkt);
  internal_memcpy(Slot + sizeof(u32), reinterpret_cast<u8*>(&Pkt) + sizeof(u32),
                  sizeof(Pkt) - sizeof(u32));
  __atomic_store_n(reinterpret_cast<u32*>(Slot),
                   u32(Pkt.VenHdr) << 16 | Pkt.Header, __ATOMIC_RELEASE);
  Api.hsa_signal_store_screlease(D.Queue->doorbell_signal, Index);
}

void Offload::Await(Device& D) {
  // The runtime may wake before the value meets the condition.
  while (Api.hsa_signal_wait_scacquire(D.Signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                       UINT64_MAX,
                                       HSA_WAIT_STATE_BLOCKED) != 0);
}

// Publish a metadata write to the location. This is done through a direct write
// from the command processor to ensure cache coherence without using a fence.
bool Offload::PublishDeviceBytes(uptr Addr, const void* Src, uptr Bytes,
                                 hsa_agent_t Agent) {
  Lock L(&OffloadMtx);
  if (!Bytes)
    return true;
  if (!Ready() || !Ib || (Addr % sizeof(u64)) || (Bytes % sizeof(u64))) {
    Report("ERROR: %s: device write rejected\n", SanitizerToolName);
    Die();
  }

  Device* Target = nullptr;
  for (Device& D : DeviceList)
    if (D.Agent.handle == Agent.handle)
      Target = &D;
  if (!Target || !Target->Queue) {
    Report("ERROR: %s: device write queue unavailable\n", SanitizerToolName);
    Die();
  }

  for (uptr Off = 0; Off < Bytes;) {
    u32* Out = static_cast<u32*>(Ib);
    u32 At = 0;
    while (Off < Bytes) {
      uptr Step = Min(Bytes - Off, kWriteChunkBytes);
      u32 Packet = static_cast<u32>(Step / sizeof(u32)) + 4;
      if (At + Packet > kIbDwords)
        break;
      const void* Part = Src ? static_cast<const u8*>(Src) + Off : nullptr;
      At += EmitWriteData(Out + At, Addr + Off, Part, Step);
      Off += Step;
    }
    CHECK_LE(At, kIbDwords);
    CHECK_LE(At, 0xFFFFFu);
    SubmitIb(*Target, At);
    Await(*Target);
  }
  return true;
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
