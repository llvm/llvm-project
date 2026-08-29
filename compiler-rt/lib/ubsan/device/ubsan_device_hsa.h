//===-- ubsan_device_hsa.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_DEVICE_HSA_H
#define UBSAN_DEVICE_HSA_H

#include "hsa.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"

#define UBSAN_HSA_LIBRARY "libhsa-runtime64"

namespace __ubsan {

#define UBSAN_HSA_FUNCTIONS(X)                                                 \
  X(hsa_iterate_agents)                                                        \
  X(hsa_agent_get_info)                                                        \
  X(hsa_executable_get_symbol_by_name)                                         \
  X(hsa_executable_symbol_get_info)                                            \
  X(hsa_memory_copy)                                                           \
  X(hsa_amd_memory_pool_allocate)                                              \
  X(hsa_amd_memory_pool_free)                                                  \
  X(hsa_amd_agents_allow_access)                                               \
  X(hsa_amd_memory_pool_get_info)                                              \
  X(hsa_amd_agent_iterate_memory_pools)                                        \
  X(hsa_system_get_major_extension_table)                                      \
  X(hsa_amd_signal_create)                                                     \
  X(hsa_signal_destroy)                                                        \
  X(hsa_signal_store_screlease)                                                \
  X(hsa_signal_wait_scacquire)

class Hsa {
public:
  bool Ready() const { return atomic_load(&Active, memory_order_acquire) != 0; }
  bool Init();
  void Shutdown();

  bool AddRef();
  bool DropRef();

  InternalMmapVectorNoCtor<hsa_agent_t> Devices;
  hsa_signal_t Doorbell;
  u64 *DoorbellValue;
  u64 *DoorbellMailbox;
  u32 DoorbellEvent;

  bool AllocFineGrained(uptr Bytes, void **Out);
  void Free(void *P);
  bool Copy(void *Dst, const void *Src, uptr N);
  const void *HostAddr(uptr Dev);
  bool SymbolAddr(hsa_executable_t Exec, const char *Name, hsa_agent_t Agent,
                  u64 *Addr);
  void RpcInfo(hsa_agent_t Agent, u32 *Lanes, u32 *Waves);

  void WaitDoorbell();
  void KickDoorbell();

  void RecordExecutable(hsa_executable_t Exec);
  void ForgetExecutable(hsa_executable_t Exec);

private:
  struct Api {
#define UBSAN_HSA_DECLARE(Name) decltype(&::Name) Name;
    UBSAN_HSA_FUNCTIONS(UBSAN_HSA_DECLARE)
#undef UBSAN_HSA_DECLARE
  } Api;

  InternalMmapVectorNoCtor<hsa_agent_t> Agents;
  InternalMmapVectorNoCtor<hsa_executable_t> Executables;
  hsa_amd_memory_pool_t FineGrainedPool;
  uptr Refs;
  atomic_uint8_t Active;

  LoaderApi Loader;

  bool Resolve();
  bool Discover();
  bool BindLoader();
  bool BindDoorbell();
  bool ExecutableInfo(hsa_loaded_code_object_t Obj,
                      hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                      u64 *Out);

  template <typename Cb> void ForEachAgentObject(hsa_executable_t Exec, Cb F);
};

Hsa &GetHsa();

} // namespace __ubsan

#endif // UBSAN_DEVICE_HSA_H
