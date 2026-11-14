//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// HSA interceptors for the device address sanitizer runtime.
///
//===----------------------------------------------------------------------===//

#include "dasan.h"
#include "dasan_allocator.h"
#include "dasan_flags.h"
#include "dasan_image.h"
#include "dasan_platform.h"
#include "dasan_report.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

#if !SANITIZER_LINUX
#  error "the device address sanitizer is supported on Linux only"
#endif

using namespace __sanitizer;
using namespace __dasan;

// The runtime exports these to every program, act as if HSA is absent when it
// is not loaded.
#define DASAN_HSA_ENTER(name)                                               \
  Initialize();                                                             \
  if (UNLIKELY(!REAL(name))) {                                              \
    INTERCEPT_FUNCTION(name);                                               \
    if (UNLIKELY(!REAL(name))) {                                            \
      VReport(1, "%s: cannot find %s in this process\n", SanitizerToolName, \
              #name);                                                       \
      return HSA_STATUS_ERROR;                                              \
    }                                                                       \
  }

// Passthrough to the real version if initialization was unsuccessful.
#define DASAN_HSA_FORWARD(name, ...)                         \
  DASAN_HSA_ENTER(name);                                     \
  if (UNLIKELY(!Offload::Get().Ready() || !PlatformReady())) \
    return REAL(name)(__VA_ARGS__);

// Bind the VA space and internal state after the offloading layer.
static void Bind() {
  Lock L(&DasanMutex);
  if (!Offload::Get().Ready() || PlatformReady())
    return;
  if (!PlatformInit() || !GetAllocator().Init()) {
    PlatformShutdown();
    return;
  }
  PlatformCommit();
}

// Release the VA reservation and tear down the internal state.
static void Unbind() {
  Lock L(&DasanMutex);
  if (!PlatformReady() || Offload::Get().Ready())
    return;
  UntrackAll();
  PrintStats();
  GetAllocator().Shutdown();
  PlatformShutdown();
}

static void WatchGpuFaults();

INTERCEPTOR(hsa_status_t, hsa_init, void) {
  DASAN_HSA_ENTER(hsa_init);

  hsa_status_t Status = REAL(hsa_init)();
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  Offload::Get().Init();
  Bind();
  if (!PlatformReady()) {
    static bool Told = false;
    if (!Told) {
      Told = true;
      Report("%s: initialization failed, runtime disabled\n",
             SanitizerToolName);
    }
  } else if (!flags()->alias_globals || !Offload::Get().CanLoadAtAddress()) {
    static bool Told = false;
    if (!Told) {
      Told = true;
      Report("%s: device globals will not be checked\n", SanitizerToolName);
    }
  }
  if (PlatformReady())
    WatchGpuFaults();
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_shut_down, void) {
  DASAN_HSA_ENTER(hsa_shut_down);

  // Drains outstanding reports while the allocator can still describe them.
  Offload::Get().Shutdown();
  Unbind();
  return REAL(hsa_shut_down)();
}

// Loads the user's executable into the tracked VA range. Global values will
// pass the membership test and be checked like heap allocations.
INTERCEPTOR(hsa_status_t, hsa_executable_load_agent_code_object,
            hsa_executable_t Executable, hsa_agent_t Agent,
            hsa_code_object_reader_t Reader, const char* Options,
            hsa_loaded_code_object_t* Loaded) {
  DASAN_HSA_FORWARD(hsa_executable_load_agent_code_object, Executable, Agent,
                    Reader, Options, Loaded);

  // The runtime does not support the necessary 1.04 loader extension. Loads the
  // executable normally outside the checked VA range.
  if (!flags()->alias_globals || !Offload::Get().CanLoadAtAddress())
    return REAL(hsa_executable_load_agent_code_object)(Executable, Agent,
                                                       Reader, Options, Loaded);

  uptr LoadSize = 0;
  if (!Offload::Get().CodeObjectLoadSize(Reader, &LoadSize))
    return REAL(hsa_executable_load_agent_code_object)(Executable, Agent,
                                                       Reader, Options, Loaded);

  // Map the image into the window. The loader copies it there on freeze.
  uptr Addr = 0, Span = 0;
  {
    Lock L(&DasanMutex);
    if (!ReserveExecutable(Agent, LoadSize, Addr, Span))
      Addr = 0;
  }
  if (!Addr)
    return REAL(hsa_executable_load_agent_code_object)(Executable, Agent,
                                                       Reader, Options, Loaded);

  // Load the executable into the tracked memory region.
  hsa_loaded_code_object_t Local = {};
  if (!Offload::Get().LoadCodeObjectAt(Executable, Agent, Reader, Options,
                                       reinterpret_cast<void*>(Addr), Span,
                                       Loaded ? Loaded : &Local)) {
    {
      Lock L(&DasanMutex);
      ReleaseExecutable(Addr);
    }
    return REAL(hsa_executable_load_agent_code_object)(Executable, Agent,
                                                       Reader, Options, Loaded);
  }

  Lock L(&DasanMutex);
  RecordExecutable(Addr);
  VReport(1, "%s: loaded %zu bytes at 0x%zx\n", SanitizerToolName, LoadSize,
          Addr);
  return HSA_STATUS_SUCCESS;
}

INTERCEPTOR(hsa_status_t, hsa_executable_freeze, hsa_executable_t Executable,
            const char* Options) {
  DASAN_HSA_FORWARD(hsa_executable_freeze, Executable, Options);

  // Traverse the loaded executable's globals and record their size in the
  // associated metadata region.
  hsa_status_t Status = REAL(hsa_executable_freeze)(Executable, Options);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  {
    Lock L(&DasanMutex);
    TrackExecutable(Executable);
  }

  // Track the executable for the RPC interface.
  Offload::Get().TrackExecutable(Executable);
  return Status;
}

INTERCEPTOR(hsa_status_t, hsa_executable_destroy, hsa_executable_t Executable) {
  DASAN_HSA_FORWARD(hsa_executable_destroy, Executable);

  // Forget the image and its associated globals.
  Offload::Get().UntrackExecutable(Executable);
  {
    Lock L(&DasanMutex);
    UntrackExecutable(Executable);
  }
  return REAL(hsa_executable_destroy)(Executable);
}

// Gets the thread ID for printing diagnostics without intercepting pthreads.
static StaticSpinMutex TidMu;
static ThreadID Tids[256];
static u32 NTids;

static u32 CompactTid() {
  const ThreadID Os = GetTid();
  SpinMutexLock L(&TidMu);
  for (u32 I = 0; I < NTids; ++I)
    if (Tids[I] == Os)
      return I;
  if (NTids == ARRAY_SIZE(Tids))
    return 0;
  Tids[NTids] = Os;
  return NTids++;
}

// Record the stack trace at the function call for later diagnostics.
static void CaptureSite(uptr Pc, uptr Bp, u32* Stack, u32* Tid) {
  BufferedStackTrace Trace;
  Trace.Unwind(Pc, Bp, nullptr, common_flags()->fast_unwind_on_malloc,
               common_flags()->malloc_context_size);
  *Stack = Trace.size ? StackDepotPut(Trace) : 0;
  *Tid = CompactTid();
}

// Allocate memory into the tracked virtual address window. Attempt to use the
// size class that best fits the allocation size, taking the next if exhausted.
static bool PlaceAlloc(hsa_amd_memory_pool_t MemoryPool, size_t Size,
                       uint32_t Flags, void** Ptr, u32 Stack, u32 Tid) {
  if (!Size)
    return false;
  Lock L(&DasanMutex);
  if (!IsValidPool(MemoryPool))
    return false;

  Allocator& A = GetAllocator();
  const uptr First = ChooseClass(Size, HeapAlign(Size));
  if (First >= kNumSizeClasses) {
    ++A.Count.TooLarge;
    return false;
  }

  for (uptr ClassId = First; ClassId < kNumSizeClasses; ++ClassId) {
    const uptr Placed =
        A.Allocate(MemoryPool, ClassId, Size, Flags, Stack, Tid);
    if (!Placed)
      continue;
    if (ClassId != First)
      ++A.Count.LargerClass;
    *Ptr = reinterpret_cast<void*>(Placed);
    return true;
  }
  ++A.Count.Exhausted;
  return false;
}

static bool PlaceFree(void* Ptr, u32 Stack, u32 Tid) {
  Lock L(&DasanMutex);
  return GetAllocator().Deallocate(reinterpret_cast<uptr>(Ptr), Stack, Tid);
}

// Allow agent access to both the allocated memory and its associated metadata.
static bool PlaceAllow(uptr Addr, const hsa_agent_t* Agents, u32 N,
                       hsa_status_t* Status) {
  Lock L(&DasanMutex);
  if (!GetAllocator().IsPlaced(Addr))
    return false;
  *Status = GetAllocator().AllowAccess(Addr, Agents, N) ? HSA_STATUS_SUCCESS
                                                        : HSA_STATUS_ERROR;
  return true;
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags,
            void** Ptr) {
  DASAN_HSA_FORWARD(hsa_amd_memory_pool_allocate, Pool, Size, Flags, Ptr);

  // PCI-e asks for point-to-point fine grain. A runtime that does not honor
  // that flag on VRAM would place a coarse copy of a fine-grain request.
  if ((Flags & HSA_AMD_MEMORY_POOL_PCIE_FLAG) &&
      !Offload::Get().FineGrainVmem()) {
    return REAL(hsa_amd_memory_pool_allocate)(Pool, Size, Flags, Ptr);
  }

  // Record the point of allocation for later diagnostics and attempt to place
  // the allocation in the tracked virtual address space.
  GET_CURRENT_PC_BP;
  u32 Stack, Tid;
  CaptureSite(pc, bp, &Stack, &Tid);
  if (PlaceAlloc(Pool, Size, Flags, Ptr, Stack, Tid))
    return HSA_STATUS_SUCCESS;

  // If we failed to the place the allocation fall back to the HSA placement.
  return REAL(hsa_amd_memory_pool_allocate)(Pool, Size, Flags, Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_free, void* Ptr) {
  DASAN_HSA_FORWARD(hsa_amd_memory_pool_free, Ptr);

  // Free the associated allocation and record.
  GET_CURRENT_PC_BP;
  u32 Stack, Tid;
  CaptureSite(pc, bp, &Stack, &Tid);
  if (PlaceFree(Ptr, Stack, Tid))
    return HSA_STATUS_SUCCESS;
  return REAL(hsa_amd_memory_pool_free)(Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_memory_allocate, hsa_region_t Region, size_t Size,
            void** Ptr) {
  DASAN_HSA_FORWARD(hsa_memory_allocate, Region, Size, Ptr);

  // ROCr uses one value for the region and the pool, borrow the handle.
  GET_CURRENT_PC_BP;
  u32 Stack, Tid;
  CaptureSite(pc, bp, &Stack, &Tid);
  hsa_amd_memory_pool_t Pool = {};
  Pool.handle = Region.handle;
  if (PlaceAlloc(Pool, Size, 0, Ptr, Stack, Tid))
    return HSA_STATUS_SUCCESS;
  return REAL(hsa_memory_allocate)(Region, Size, Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_memory_free, void* Ptr) {
  DASAN_HSA_FORWARD(hsa_memory_free, Ptr);

  GET_CURRENT_PC_BP;
  u32 Stack, Tid;
  CaptureSite(pc, bp, &Stack, &Tid);
  if (PlaceFree(Ptr, Stack, Tid))
    return HSA_STATUS_SUCCESS;
  return REAL(hsa_memory_free)(Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_agents_allow_access, uint32_t NumAgents,
            const hsa_agent_t* Agents, const uint32_t* Flags, const void* Ptr) {
  DASAN_HSA_FORWARD(hsa_amd_agents_allow_access, NumAgents, Agents, Flags, Ptr);

  hsa_status_t Status;
  if (PlaceAllow(reinterpret_cast<uptr>(Ptr), Agents, NumAgents, &Status))
    return Status;
  return REAL(hsa_amd_agents_allow_access)(NumAgents, Agents, Flags, Ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_set_access, void* Va, size_t Size,
            const hsa_amd_memory_access_desc_t* Desc, size_t DescCnt) {
  DASAN_HSA_FORWARD(hsa_amd_vmem_set_access, Va, Size, Desc, DescCnt);

  hsa_status_t Status;
  if (!Desc || !DescCnt) {
    if (PlaceAllow(reinterpret_cast<uptr>(Va), nullptr, 0, &Status))
      return HSA_STATUS_ERROR;
    return REAL(hsa_amd_vmem_set_access)(Va, Size, Desc, DescCnt);
  }

  InternalMmapVector<hsa_agent_t> Agents;
  for (size_t I = 0; I < DescCnt; ++I) Agents.push_back(Desc[I].agent_handle);
  if (PlaceAllow(reinterpret_cast<uptr>(Va), Agents.data(),
                 static_cast<u32>(Agents.size()), &Status))
    return Status;
  return REAL(hsa_amd_vmem_set_access)(Va, Size, Desc, DescCnt);
}

INTERCEPTOR(hsa_status_t, hsa_amd_pointer_info, const void* Ptr,
            hsa_amd_pointer_info_t* Info, void* (*Alloc)(size_t),
            uint32_t* NumAgents, hsa_agent_t** Accessible) {
  DASAN_HSA_FORWARD(hsa_amd_pointer_info, Ptr, Info, Alloc, NumAgents,
                    Accessible);

  // The runtime keeps track of virtual memory allocations. These must be
  // modified to appear as if they came from a standard allocation.
  hsa_status_t Status =
      REAL(hsa_amd_pointer_info)(Ptr, Info, Alloc, NumAgents, Accessible);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;

  AllocInfo A;
  u32 Flags = 0;
  hsa_agent_t Owner = {};
  bool HaveOwner = false;
  {
    Lock L(&DasanMutex);
    if (!GetAllocator().getAllocInfo(reinterpret_cast<uptr>(Ptr), &A) ||
        A.Freed)
      return Status;
    Flags = PoolFlags(A.Pool);
    HaveOwner = AgentForPool(A.Pool, &Owner);
  }

  const uint32_t Size = Info->size;
  if (Size < offsetof(hsa_amd_pointer_info_t, sizeInBytes) + sizeof(size_t))
    return Status;
  Info->type = HSA_EXT_POINTER_TYPE_HSA;
  Info->agentBaseAddress = reinterpret_cast<void*>(A.Beg);
  Info->hostBaseAddress = Info->agentBaseAddress;
  Info->sizeInBytes = A.Size;
  if (HaveOwner && Size >= offsetof(hsa_amd_pointer_info_t, agentOwner) +
                               sizeof(Info->agentOwner))
    Info->agentOwner = Owner;
  if (Size >= offsetof(hsa_amd_pointer_info_t, global_flags) +
                  sizeof(Info->global_flags))
    Info->global_flags = Flags;
  return Status;
}

// A non-success return leaves the runtime's own fault report in place.
static hsa_status_t OnGpuEvent(const hsa_amd_event_t* Event, void*) {
  if (Event && Event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT &&
      PointerIsMetadata(Event->memory_fault.virtual_address))
    Report(
        "%s: GPU fault at 0x%zx is inside metadata. The program likely formed "
        "an illegal out-of-bounds base pointer.\n",
        SanitizerToolName,
        static_cast<uptr>(Event->memory_fault.virtual_address));
  return HSA_STATUS_ERROR;
}

// The wrapper exists so REAL() has a type. Registration itself is unchanged.
INTERCEPTOR(hsa_status_t, hsa_amd_register_system_event_handler,
            hsa_amd_system_event_callback_t Callback, void* Data) {
  DASAN_HSA_FORWARD(hsa_amd_register_system_event_handler, Callback, Data);
  return REAL(hsa_amd_register_system_event_handler)(Callback, Data);
}

// The runtime blindly trusts 'inbounds' pointers. Creating a base pointer
// outside of the allocation range is undefined behavior, but manifests as a
// memory fault in the metadata arena. Watch the faults to hint to the user.
static void WatchGpuFaults() {
  static bool Done = false;
  if (Done)
    return;
  Done = true;
  if (!REAL(hsa_amd_register_system_event_handler))
    INTERCEPT_FUNCTION(hsa_amd_register_system_event_handler);
  if (!REAL(hsa_amd_register_system_event_handler) ||
      REAL(hsa_amd_register_system_event_handler)(OnGpuEvent, nullptr) !=
          HSA_STATUS_SUCCESS)
    VReport(1, "%s: GPU fault watch unavailable\n", SanitizerToolName);
}

[[gnu::constructor(0)]] static void DasanDynInit() {
  __dasan_init();
  WarnIfHsaNotInterposed();
}

[[gnu::destructor]] static void DasanAtExit() {
  Lock L(&DasanMutex);
  PrintStats();
}
