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
  SymbolizedStack* Symbolize(uptr PC);

 private:
  friend struct OffloadRpc;

  constexpr Offload() = default;

  static Offload Ctx;

  struct Device {
    hsa_agent_t Agent;
    hsa_amd_memory_pool_t Pool;
    u32 Lanes;
    u32 MaxWaves;
  };

  struct Api {
#define SANITIZER_HSA_DECLARE(Name) decltype(&::Name) Name;
    SANITIZER_HSA_FUNCTIONS(SANITIZER_HSA_DECLARE)
#undef SANITIZER_HSA_DECLARE
  } Api{};

  bool Resolve();
  bool Discover();
  bool BindLoader();
  bool Release();
  void Teardown();

  const Device& Host() const { return HostDevice; }
  const InternalMmapVectorNoCtor<Device>& Devices() const { return DeviceList; }

  bool Alloc(const Device& D, uptr Bytes, void** Out);
  void Free(void* P);
  bool Copy(void* Dst, const void* Src, uptr N);
  bool Lookup(hsa_executable_t Exec, const char* Name, hsa_agent_t Agent,
              u64* Addr);
  bool CreateSignal(hsa_signal_t* Out);
  void DestroySignal(hsa_signal_t Sig);
  void WaitSignal(hsa_signal_t Sig);
  void StoreSignal(hsa_signal_t Sig);

  void TrackImage(uptr LoadBase, uptr LoadSize, const void* Storage,
                  uptr StorageSize);
  void UntrackImage(uptr LoadBase);
  bool ExecutableInfo(hsa_loaded_code_object_t Obj,
                      hsa_ven_amd_loader_loaded_code_object_info_t Attr,
                      u64* Out);
  template <typename Cb>
  void ForEachAgentObject(hsa_executable_t Exec, Cb F);

  Device HostDevice{};
  InternalMmapVectorNoCtor<Device> DeviceList{};

  // Serializes initialization and shutdown of the core HSA interface.
  Mutex LifetimeMtx{};

  // Serializes HSA functions that affect the common compiler-rt state.
  Mutex OffloadMtx{};
  uptr Refs{};
  atomic_uint8_t Active{};
  LoaderApi Loader{};
};

}  // namespace __sanitizer

#endif  // SANITIZER_OFFLOAD_H
