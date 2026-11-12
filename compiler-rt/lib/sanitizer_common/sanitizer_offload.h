//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host GPU shim for sanitizer runtimes. HSA is resolved, never linked.
// Callers must not include a system hsa.h alongside this header.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_OFFLOAD_H
#define SANITIZER_OFFLOAD_H

#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
#include "sanitizer_mutex.h"
#include "sanitizer_offload_hsa.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

struct OffloadRpc;

class Offload {
 public:
  static Offload& Get();

  bool Init();
  void Shutdown();
  bool Ready() const;

  const void* HostPointer(uptr DeviceAddr);

  using Handler = u32 (*)(void* Port, u32 Lanes);
  void RegisterHandler(Handler Fn);

  void TrackExecutable(hsa_executable_t Exec);
  void UntrackExecutable(hsa_executable_t Exec);
  void UntrackImages();
  bool GetMemoryPool(hsa_agent_t Agent, hsa_amd_memory_pool_t* Pool);
  bool Allocate(hsa_amd_memory_pool_t Pool, uptr Bytes, void** Out);
  SymbolizedStack* Symbolize(uptr PC);
  bool SymbolizeData(uptr Addr, DataInfo* Info);

  // Informating relating to an HSA memory pool.
  struct MemoryPool {
    hsa_agent_t Agent;
    hsa_amd_memory_pool_t Pool;
    u32 Flags;
    uptr Granule;
    // Minimum the pool accepts. Granule may be the larger recommended block.
    uptr AllocGranule;
    bool HostAccess;
  };

  struct CodeObject {
    hsa_agent_t Agent;
    uptr LoadBase;
    uptr LoadSize;
    const void* Storage;
    uptr StorageSize;
  };

  bool IsGpu(hsa_agent_t Agent) const;
  const InternalMmapVectorNoCtor<hsa_agent_t>& Cpus() const { return CpuList; }
  const InternalMmapVectorNoCtor<MemoryPool>& Pools() const { return PoolList; }

  // Pool on Agent whose global flags intersect Match. Coarse-grain wins when
  // several match.
  bool PoolForAgent(hsa_agent_t Agent, u32 Match, MemoryPool* Out);
  // Discovered record for this pool.
  bool FindPool(hsa_amd_memory_pool_t Pool, MemoryPool* Out);
  // First fine-grain pool on a non-GPU agent.
  bool HostPool(MemoryPool* Out);

  // Copies the discovered pools, then runs F without OffloadMtx held.
  template <typename Fn>
  void ForEachPool(Fn F) {
    InternalMmapVector<MemoryPool> Copy;
    {
      Lock L(&OffloadMtx);
      if (!Ready())
        return;
      for (const MemoryPool& P : PoolList) Copy.push_back(P);
    }
    for (const MemoryPool& P : Copy) F(P);
  }

  bool CanAccess(hsa_agent_t Agent, hsa_amd_memory_pool_t Pool) const;
  bool Copy(void* Dst, const void* Src, uptr N) const;

  bool FineGrainVmem() const;
  bool ReserveAddress(uptr Addr, uptr Size);
  void FreeAddress(uptr Addr, uptr Size);
  bool CreateHandle(hsa_amd_memory_pool_t Pool, uptr Size, u64 Flags,
                    u64* Handle);
  void ReleaseHandle(u64 Handle);
  bool Map(uptr Addr, uptr Size, uptr Offset, u64 Handle);
  void Unmap(uptr Addr, uptr Size);
  bool SetAccess(uptr Addr, uptr Size, const hsa_amd_memory_access_desc_t* Desc,
                 uptr Count);

  // Loader 1.04 can place a code object at a reserved address.
  bool CanLoadAtAddress() const;
  bool CodeObjectLoadSize(hsa_code_object_reader_t Reader, uptr* Size) const;
  bool LoadCodeObjectAt(hsa_executable_t Exec, hsa_agent_t Agent,
                        hsa_code_object_reader_t Reader, const char* Options,
                        void* Address, uptr Size,
                        hsa_loaded_code_object_t* Loaded) const;

  // The callback runs without the offload lock held.
  template <typename Fn>
  void ForEachCodeObject(hsa_executable_t Exec, Fn F) {
    VisitCodeObjects(
        Exec,
        [](const CodeObject& Obj, void* Arg) { (*static_cast<Fn*>(Arg))(Obj); },
        &F);
  }

  // Confirmed L2 store of [Addr, Addr + Bytes) through the GPU that owns
  // the pages. Src null stores zeros.
  bool PublishDeviceBytes(uptr Addr, const void* Src, uptr Bytes,
                          hsa_agent_t Agent);

 private:
  friend struct OffloadRpc;

  constexpr Offload() = default;

  static Offload Ctx;

  struct Device {
    hsa_agent_t Agent;
    u32 Lanes;
    u32 MaxWaves;
    hsa_queue_t* Queue;
    hsa_signal_t Signal;
  };

  struct Api {
#define SANITIZER_HSA_DECLARE(Name) decltype(&::Name) Name;
    SANITIZER_HSA_FUNCTIONS(SANITIZER_HSA_DECLARE)
#undef SANITIZER_HSA_DECLARE
  } Api{};

  bool Resolve();
  bool Discover();
  bool BindLoader();
  bool CreateCommands();
  bool Release();
  void Teardown();

  const InternalMmapVectorNoCtor<Device>& Devices() const { return DeviceList; }

  bool IsCpu(hsa_agent_t Agent) const;
  bool Alloc(const MemoryPool& Pool, uptr Bytes, void** Out, u64 Flags = 0);
  void Free(void* P);
  bool Lookup(hsa_executable_t Exec, const char* Name, hsa_agent_t Agent,
              u64* Addr);
  bool CreateSignal(hsa_signal_t* Out);
  void DestroySignal(hsa_signal_t Sig);
  void WaitSignal(hsa_signal_t Sig);
  void StoreSignal(hsa_signal_t Sig);

  void SubmitIb(Device& D, u32 Dwords);
  void Await(Device& D);

  void TrackImage(uptr LoadBase, uptr LoadSize, const void* Storage,
                  uptr StorageSize);
  void UntrackImage(uptr LoadBase);
  bool ExecutableInfo(hsa_loaded_code_object_t Obj,
                      hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                      u64* Out);
  bool DescribeCodeObject(hsa_loaded_code_object_t Obj, hsa_agent_t Agent,
                          CodeObject* Out);
  template <typename Cb>
  void ForEachAgentObject(hsa_executable_t Exec, Cb F);
  void VisitCodeObjects(hsa_executable_t Exec,
                        void (*Fn)(const CodeObject&, void*), void* Arg);

  u16 AmdExtMinor{};
  void* Ib{};

  InternalMmapVectorNoCtor<Device> DeviceList{};
  InternalMmapVectorNoCtor<hsa_agent_t> CpuList{};
  InternalMmapVectorNoCtor<MemoryPool> PoolList{};

  // Serializes initialization and shutdown of the core HSA interface.
  Mutex LifetimeMtx{};

  // Serializes HSA functions that affect the common compiler-rt state.
  Mutex OffloadMtx{};
  uptr Refs{};
  atomic_uint8_t Active{};
  LoaderApi Loader{};
};

void WarnIfHsaNotInterposed();

}  // namespace __sanitizer

#endif  // SANITIZER_OFFLOAD_H
