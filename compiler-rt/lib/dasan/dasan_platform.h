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

#ifndef DASAN_PLATFORM_H
#define DASAN_PLATFORM_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_offload_hsa.h"

namespace __dasan {

using namespace __sanitizer;

inline bool operator==(hsa_amd_memory_pool_t A, hsa_amd_memory_pool_t B) {
  return A.handle == B.handle;
}
inline bool operator!=(hsa_amd_memory_pool_t A, hsa_amd_memory_pool_t B) {
  return !(A == B);
}
// A zero handle is the host pool, resolved when a handle is created.
inline bool HasPool(hsa_amd_memory_pool_t Pool) { return Pool.handle != 0; }

bool PlatformInit();
void PlatformShutdown();
// The VA window is reserved. Callers hold DasanMutex.
void PlatformCommit();
bool PlatformReady();

// One placeable pool per GPU, coarse preferred, in agent order.
void PlaceablePools(InternalMmapVector<hsa_amd_memory_pool_t>* Out);
// True when this pool is one we place into.
bool IsValidPool(hsa_amd_memory_pool_t Pool);
hsa_amd_memory_pool_t PoolForAgent(hsa_agent_t Agent);
bool AgentForPool(hsa_amd_memory_pool_t Pool, hsa_agent_t* Out);
u32 PoolFlags(hsa_amd_memory_pool_t Pool);

uptr VaGranule();
// Allocation granule for metadata handles. VaGranule() is the recommended
// block.
uptr VaMetaGranule();

bool VaReserve(uptr Addr, uptr Size);
void VaRelease(uptr Addr, uptr Size);

bool VaCreate(hsa_amd_memory_pool_t Pool, uptr Size, u32 Flags, u64* Handle);
void VaDestroy(u64 Handle);

bool VaMap(uptr Addr, uptr Size, u64 Handle, hsa_amd_memory_pool_t Pool,
           uptr Offset = 0, bool Executable = false);
void VaUnmap(uptr Addr, uptr Size);
bool VaAllow(uptr Addr, uptr Size, hsa_amd_memory_pool_t Pool);
bool VaAllowAgents(uptr Addr, uptr Size, const hsa_agent_t* Agents, u32 N);

bool VaProbe(uptr Addr);
// Store [Addr, Addr + N). A GPU-owned range goes through that agent's
// command processor and is confirmed in L2. A host-owned range is stored
// directly. Null Src stores zeros. A failed store is fatal.
void VaPublish(uptr Addr, const void* Src, uptr N);

}  // namespace __dasan

#endif  // DASAN_PLATFORM_H
