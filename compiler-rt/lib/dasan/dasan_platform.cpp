//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// How each reserved range is mapped, and who may touch it. Which pool backs a
/// range is answered by the offload list.
///
//===----------------------------------------------------------------------===//

#include "dasan_platform.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_offload.h"

using namespace __sanitizer;

namespace __dasan {
namespace {

constexpr uptr kNpos = ~(uptr)0;

template <typename Vec, typename Pred>
uptr FindIndex(const Vec& V, Pred P) {
  for (uptr I = 0; I < V.size(); ++I)
    if (P(V[I]))
      return I;
  return kNpos;
}

struct MappedRange {
  uptr Addr;
  uptr Size;
  hsa_amd_memory_pool_t Pool;
  bool CpuRw;

  bool Covers(uptr P) const { return P >= Addr && P < Addr + Size; }
  bool Overlaps(uptr Beg, uptr End) const {
    return Addr < End && Beg < Addr + Size;
  }
};

struct State {
  InternalMmapVectorNoCtor<MappedRange> Mapped;
  uptr Granule;
  uptr MetaGranule;
  atomic_uint8_t Bound;
};
State S;

Offload& O() { return Offload::Get(); }

// The memory types we can correct place and track in the reserved VA region.
u32 PlaceableMask() {
  u32 Match = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  if (O().FineGrainVmem())
    Match |= HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
             HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED;
  return Match;
}

bool Lookup(hsa_amd_memory_pool_t Pool, Offload::MemoryPool* Out) {
  return HasPool(Pool) && O().FindPool(Pool, Out);
}

bool HostCanAccess(hsa_amd_memory_pool_t Pool) {
  Offload::MemoryPool Found;
  if (!HasPool(Pool))
    return O().HostPool(&Found);
  return Lookup(Pool, &Found) && Found.HostAccess;
}

bool CpuCanAccess(hsa_agent_t Cpu, hsa_amd_memory_pool_t Pool) {
  return !HasPool(Pool) || O().CanAccess(Cpu, Pool);
}

bool CpuMapped(uptr Addr) {
  const uptr I =
      FindIndex(S.Mapped, [&](const MappedRange& R) { return R.Covers(Addr); });
  return I != kNpos && S.Mapped[I].CpuRw;
}

void DropMapped(uptr I) {
  if (I + 1 != S.Mapped.size())
    S.Mapped[I] = S.Mapped.back();
  S.Mapped.pop_back();
}

// The host stays writable when it can reach the pages. The owning GPU gets
// read-write, or read-write-execute for a code object.
bool Grant(uptr Addr, uptr Size, hsa_amd_memory_pool_t Pool, bool WithCpu,
           bool Executable) {
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  hsa_agent_t Owner = {};
  const bool HaveOwner = HasPool(Pool) && AgentForPool(Pool, &Owner);
  if (WithCpu) {
    for (hsa_agent_t Cpu : O().Cpus()) {
      if (!CpuCanAccess(Cpu, Pool))
        continue;
      Descs.push_back({HSA_ACCESS_PERMISSION_RW, Cpu});
    }
  }
  if (HaveOwner)
    Descs.push_back(
        {Executable ? HSA_ACCESS_PERMISSION_RWX : HSA_ACCESS_PERMISSION_RW,
         Owner});
  if (Descs.empty())
    return false;
  return O().SetAccess(Addr, Size, Descs.data(), Descs.size());
}

}  // namespace

bool PlatformInit() {
  if (atomic_load(&S.Bound, memory_order_acquire))
    return true;
  Offload& Off = O();
  if (!Off.Ready())
    return false;

  S.Granule = S.MetaGranule = 0;
  const u32 Match = PlaceableMask();
  uptr N = 0;
  uptr Granule = 0;
  uptr Meta = 0;
  bool CpuDirect = true;
  Off.ForEachPool([&](const Offload::MemoryPool& P) {
    if (!Off.IsGpu(P.Agent) || !(P.Flags & Match))
      return;
    ++N;
    Granule = Max(Granule, P.Granule);
    Meta = Max(Meta, P.AllocGranule ? P.AllocGranule : P.Granule);
    CpuDirect &= P.HostAccess;
  });

  if (!N || !Granule || !Meta) {
    VReport(1, "%s: no device with placeable memory; HSA calls pass through\n",
            SanitizerToolName);
    return false;
  }
  S.Granule = Granule;
  S.MetaGranule = Meta;
  VReport(1, "%s: %zu placeable pools, granule %zu, metadata %zu%s%s\n",
          SanitizerToolName, N, S.Granule, S.MetaGranule,
          CpuDirect ? "" : ", host copies metadata",
          Off.FineGrainVmem() ? ", fine-grain vmem" : "");
  return true;
}

void PlatformShutdown() {
  atomic_store(&S.Bound, 0, memory_order_release);
  S.Mapped.clear();
  S.Granule = S.MetaGranule = 0;
}

void PlatformCommit() { atomic_store(&S.Bound, 1, memory_order_release); }

bool PlatformReady() {
  return atomic_load(&S.Bound, memory_order_acquire) != 0;
}

void PlaceablePools(InternalMmapVector<hsa_amd_memory_pool_t>* Out) {
  if (!Out)
    return;
  const u32 Match = PlaceableMask();
  InternalMmapVector<hsa_agent_t> Seen;
  O().ForEachPool([&](const Offload::MemoryPool& P) {
    if (!O().IsGpu(P.Agent))
      return;
    for (hsa_agent_t Agent : Seen)
      if (Agent.handle == P.Agent.handle)
        return;
    Seen.push_back(P.Agent);
    Offload::MemoryPool Chosen;
    if (O().PoolForAgent(P.Agent, Match, &Chosen))
      Out->push_back(Chosen.Pool);
  });
}

bool IsValidPool(hsa_amd_memory_pool_t Pool) {
  Offload::MemoryPool Found;
  return Lookup(Pool, &Found) && O().IsGpu(Found.Agent) &&
         (Found.Flags & PlaceableMask());
}

hsa_amd_memory_pool_t PoolForAgent(hsa_agent_t Agent) {
  Offload::MemoryPool Found;
  if (!O().PoolForAgent(Agent, PlaceableMask(), &Found))
    return {};
  return Found.Pool;
}

bool AgentForPool(hsa_amd_memory_pool_t Pool, hsa_agent_t* Out) {
  Offload::MemoryPool Found;
  if (!Lookup(Pool, &Found))
    return false;
  *Out = Found.Agent;
  return true;
}

u32 PoolFlags(hsa_amd_memory_pool_t Pool) {
  Offload::MemoryPool Found;
  return Lookup(Pool, &Found) ? Found.Flags : 0;
}

uptr VaGranule() { return S.Granule; }

uptr VaMetaGranule() { return S.MetaGranule; }

bool VaReserve(uptr Addr, uptr Size) { return O().ReserveAddress(Addr, Size); }

void VaRelease(uptr Addr, uptr Size) { O().FreeAddress(Addr, Size); }

bool VaCreate(hsa_amd_memory_pool_t Pool, uptr Size, u32 Flags, u64* Handle) {
  Offload::MemoryPool Found;
  const bool Ok = HasPool(Pool) ? Lookup(Pool, &Found) : O().HostPool(&Found);
  return Ok && HasPool(Found.Pool) &&
         O().CreateHandle(Found.Pool, Size, Flags, Handle);
}

void VaDestroy(u64 Handle) { O().ReleaseHandle(Handle); }

bool VaMap(uptr Addr, uptr Size, u64 Handle, hsa_amd_memory_pool_t Pool,
           uptr Offset, bool Executable) {
  if (!O().Map(Addr, Size, Offset, Handle))
    return false;
  const bool WithCpu = HostCanAccess(Pool);
  if (!Grant(Addr, Size, Pool, WithCpu, Executable)) {
    O().Unmap(Addr, Size);
    return false;
  }
  S.Mapped.resize(S.Mapped.size() + 1);
  MappedRange& R = S.Mapped.back();
  R.Addr = Addr;
  R.Size = Size;
  R.Pool = Pool;
  R.CpuRw = WithCpu;
  return true;
}

void VaUnmap(uptr Addr, uptr Size) {
  O().Unmap(Addr, Size);
  const uptr I = FindIndex(S.Mapped, [&](const MappedRange& M) {
    return M.Addr == Addr && M.Size == Size;
  });
  if (I != kNpos)
    DropMapped(I);
}

bool VaAllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N) {
  if (!N)
    return true;
  const uptr End = Addr + Size;
  InternalMmapVector<hsa_amd_memory_access_desc_t> Descs;
  for (u32 I = 0; I < N; ++I)
    Descs.push_back({HSA_ACCESS_PERMISSION_RW, Agents[I]});
  bool Hit = false;
  for (MappedRange& M : S.Mapped) {
    if (!M.Overlaps(Addr, End))
      continue;
    Hit = true;
    if (!O().SetAccess(M.Addr, M.Size, Descs.data(), Descs.size()))
      return false;
  }
  return Hit;
}

bool VaAllow(uptr Addr, uptr Size, hsa_amd_memory_pool_t Pool) {
  hsa_agent_t Agent = {};
  if (!AgentForPool(Pool, &Agent))
    return false;
  return VaAllowAgents(Addr, Size, &Agent, 1);
}

bool VaProbe(uptr Addr) {
  const u64 Magic = 0xda5a4e0000da5a4eULL;
  u64 Got = 0;
  if (CpuMapped(Addr)) {
    *reinterpret_cast<volatile u64*>(Addr) = Magic;
    Got = *reinterpret_cast<volatile u64*>(Addr);
    return Got == Magic;
  }
  void* Dst = reinterpret_cast<void*>(Addr);
  return O().Copy(Dst, &Magic, sizeof(Magic)) &&
         O().Copy(&Got, Dst, sizeof(Got)) && Got == Magic;
}

void VaPublish(uptr Addr, const void* Src, uptr N) {
  if (!N)
    return;
  uptr Done = 0;
  while (Done < N) {
    uptr Pos = Addr + Done;
    const uptr I = FindIndex(
        S.Mapped, [&](const MappedRange& R) { return R.Covers(Pos); });
    if (I == kNpos) {
      Report("ERROR: %s: device write has no mapping at 0x%zx\n",
             SanitizerToolName, Pos);
      Die();
    }
    const MappedRange& M = S.Mapped[I];
    uptr Step = Min(N - Done, M.Addr + M.Size - Pos);
    const void* Part = Src ? static_cast<const u8*>(Src) + Done : nullptr;
    // Pages belong to the pool's agent. A GPU owner is stored through its
    // command processor; the host is stored directly. Access is not ownership.
    hsa_agent_t Owner = {};
    if (HasPool(M.Pool) && AgentForPool(M.Pool, &Owner) && O().IsGpu(Owner) &&
        !O().PublishDeviceBytes(Pos, Part, Step, Owner)) {
      Report("ERROR: %s: device write failed at 0x%zx\n", SanitizerToolName,
             Pos);
      Die();
    }
    // The host check loads this mapping. The confirmed L2 store does not
    // update the copy that load reads.
    if (M.CpuRw) {
      void* Dst = reinterpret_cast<void*>(Pos);
      if (Part)
        internal_memcpy(Dst, Part, Step);
      else
        internal_memset(Dst, 0, Step);
    }
    Done += Step;
  }
}

}  // namespace __dasan
