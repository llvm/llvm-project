//===-- sanitizer_allocator_amdgpu.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Sanitizer Allocator.
//
//===----------------------------------------------------------------------===//
#if SANITIZER_AMDHSA
#  include <dlfcn.h>  // For dlopen, dlsym

#  include "sanitizer_allocator.h"
#  include "sanitizer_atomic.h"

namespace __sanitizer {
struct HsaFunctions {
  // -------------- Memory Allocate/Deallocate Functions ----------------
  hsa_status_t (*memory_pool_allocate)(hsa_amd_memory_pool_t memory_pool,
                                       size_t size, uint32_t flags, void** ptr);
  hsa_status_t (*memory_pool_free)(void* ptr);
  hsa_status_t (*pointer_info)(void* ptr, hsa_amd_pointer_info_t* info,
                               void* (*alloc)(size_t),
                               uint32_t* num_agents_accessible,
                               hsa_agent_t** accessible);
  hsa_status_t (*vmem_address_reserve_align)(void** ptr, size_t size,
                                             uint64_t address,
                                             uint64_t alignment,
                                             uint64_t flags);
  hsa_status_t (*vmem_address_free)(void* ptr, size_t size);

  // ----------------- System Event Register Function -------------------
  hsa_status_t (*register_system_event_handler)(
      hsa_amd_system_event_callback_t callback, void* data);
};

static HsaFunctions hsa_amd;

// ROCr handle
static void* hsa_runtime_handle = nullptr;
// Mutex to protect the ROCr handle
static StaticSpinMutex hsa_runtime_init_mu;

// Default ROCr library path
static const char kDefaultHsaRuntimePath[] = "libhsa-runtime64.so";

// Always align to page boundary to match current ROCr behavior
static const size_t kPageSize_ = 4096;

static atomic_uint8_t amdgpu_runtime_shutdown{0};
static atomic_uint8_t amdgpu_event_registered{0};
static atomic_uint8_t hsa_amd_symbols_loaded{0};

static bool HsaSymbolsLoaded() {
  return atomic_load(&hsa_amd_symbols_loaded, memory_order_acquire) != 0;
}

bool AmdgpuDeviceAllocator::IsRuntimeShutdown() {
  return static_cast<bool>(
      atomic_load(&amdgpu_runtime_shutdown, memory_order_acquire));
}

void AmdgpuDeviceAllocator::NotifyRuntimeShutdown() {
  uint8_t shutdown = 0;
  if (atomic_compare_exchange_strong(&amdgpu_runtime_shutdown, &shutdown, 1,
                                     memory_order_acq_rel)) {
    VReport(2, "Amdgpu Allocator: AMDGPU runtime shutdown detected\n");
  }
}

// Clear shutdown state when hsa_init() succeeds again (re-init after shutdown).
// Resets amdgpu_runtime_shutdown so allocator operations are enabled, and
// amdgpu_event_registered so RegisterSystemEventHandlers() will register the
// shutdown callback for the new runtime instance.
void AmdgpuDeviceAllocator::ClearRuntimeShutdownState() {
  atomic_store(&amdgpu_runtime_shutdown, 0, memory_order_release);
  atomic_store(&amdgpu_event_registered, 0, memory_order_release);
}

void AmdgpuDeviceAllocator::NoteDeviceAllocatorFailure(
    DeviceAllocationInfo* da_info, DeviceAllocFailure failure) {
  if (!da_info || da_info->type_ != DAT_AMDGPU)
    return;
  AmdgpuAllocationInfo* aa_info =
      reinterpret_cast<AmdgpuAllocationInfo*>(da_info);
  switch (failure) {
    case DEV_ALLOC_FAILURE_NOT_INITIALIZED:
      aa_info->EnsureFailureStatus(HSA_STATUS_ERROR_NOT_INITIALIZED);
      break;
    case DEV_ALLOC_FAILURE_OUT_OF_RESOURCES:
      aa_info->EnsureFailureStatus(HSA_STATUS_ERROR_OUT_OF_RESOURCES);
      break;
  }
}

// Init(false): no-op during __asan_init (no dlopen, no dlsym).
// Init(true): from asan_hsa_init() after REAL(hsa_init); ROCr may be up while
// this still fails (hsa_init can succeed with an unloaded device backend).
// Bind hsa_amd_* via dlopen(RTLD_NOLOAD) + dlsym(handle) — not REAL() here
// and not RTLD_NEXT/DEFAULT (those resolve ASan interceptors and recurse).
bool AmdgpuDeviceAllocator::Init(bool allow_dlopen) {
  SpinMutexLock l(&hsa_runtime_init_mu);
  if (HsaSymbolsLoaded())
    return true;

  // Never dlopen during __asan_init.
  if (!allow_dlopen)
    return false;

  if (!hsa_runtime_handle) {
    const char* path = GetEnv("SANITIZER_HSA_RUNTIME_PATH");
    if (!path || !path[0])
      path = kDefaultHsaRuntimePath;
    // Prefer first `RTLD_NOLOAD` when ROCr is already mapped(after successful
    // REAL(hsa_init)). If RTLD_NOLOAD fails, try `RTLD_NOW` (when ROCr is not
    // loaded).
    hsa_runtime_handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (!hsa_runtime_handle)
      hsa_runtime_handle = dlopen(path, RTLD_NOW);
    if (!hsa_runtime_handle) {
      const char* err = dlerror();
      VReport(2, "Amdgpu Init: dlopen(%s) failed: %s\n", path,
              err ? err : "unknown error");
      return false;
    }
  }

  struct HsaSymbolEntry {
    const char* name;
    void** slot;
  };
  // HSA Symbol Entries Table
  const HsaSymbolEntry kSymbols[] = {
      {"hsa_amd_memory_pool_allocate", (void**)&hsa_amd.memory_pool_allocate},
      {"hsa_amd_memory_pool_free", (void**)&hsa_amd.memory_pool_free},
      {"hsa_amd_pointer_info", (void**)&hsa_amd.pointer_info},
      {"hsa_amd_vmem_address_reserve_align",
       (void**)&hsa_amd.vmem_address_reserve_align},
      {"hsa_amd_vmem_address_free", (void**)&hsa_amd.vmem_address_free},
      {"hsa_amd_register_system_event_handler",
       (void**)&hsa_amd.register_system_event_handler},
  };

  bool success = true;
  for (uptr i = 0; i < ARRAY_SIZE(kSymbols); ++i) {
    void* sym = dlsym(hsa_runtime_handle, kSymbols[i].name);
    if (!sym) {
      VReport(2, "Amdgpu Init: Failed to load %s from ROCr handle\n",
              kSymbols[i].name);
      success = false;
    }
    *kSymbols[i].slot = sym;
  }
  if (!success) {
    internal_memset(&hsa_amd, 0, sizeof(hsa_amd));
    VReport(1, "Amdgpu Init: Failed to load AMDGPU runtime functions\n");
    return false;
  }
  atomic_store(&hsa_amd_symbols_loaded, 1, memory_order_release);
  return true;
}

void* AmdgpuDeviceAllocator::Allocate(uptr size, uptr alignment,
                                      DeviceAllocationInfo* da_info) {
  AmdgpuAllocationInfo* aa_info =
      reinterpret_cast<AmdgpuAllocationInfo*>(da_info);

  // Do not allocate if AMDGPU runtime is shutdown
  if (UNLIKELY(IsRuntimeShutdown())) {
    VReport(1,
            "Amdgpu Allocate: Runtime shutdown, skipping allocation for size "
            "%zu alignment %zu\n",
            size, alignment);
    aa_info->EnsureFailureStatus(HSA_STATUS_ERROR_INVALID_RUNTIME_STATE);
    return nullptr;
  }

  if (!HsaSymbolsLoaded()) {
    aa_info->EnsureFailureStatus(HSA_STATUS_ERROR_NOT_INITIALIZED);
    return nullptr;
  }

  if (!aa_info->memory_pool.handle) {
    aa_info->status = hsa_amd.vmem_address_reserve_align(
        &aa_info->ptr, size, aa_info->address, aa_info->alignment,
        aa_info->flags64);
  } else {
    aa_info->status = hsa_amd.memory_pool_allocate(
        aa_info->memory_pool, size, aa_info->flags, &aa_info->ptr);
  }
  if (aa_info->status != HSA_STATUS_SUCCESS)
    return nullptr;

  return aa_info->ptr;
}

void AmdgpuDeviceAllocator::Deallocate(void* p) {
  // Deallocate does nothing after AMDGPU runtime shutdown
  if (UNLIKELY(IsRuntimeShutdown())) {
    VReport(
        1,
        "Amdgpu Deallocate: Runtime shutdown, skipping deallocation for %p\n",
        reinterpret_cast<void*>(p));
    return;
  }

  DevicePointerInfo DevPtrInfo;
  if (AmdgpuDeviceAllocator::GetPointerInfo(reinterpret_cast<uptr>(p),
                                            &DevPtrInfo)) {
    if (DevPtrInfo.type == HSA_EXT_POINTER_TYPE_HSA) {
      UNUSED hsa_status_t status = hsa_amd.memory_pool_free(p);
    } else if (DevPtrInfo.type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR) {
      UNUSED hsa_status_t status =
          hsa_amd.vmem_address_free(p, DevPtrInfo.map_size);
    }
  }
}

static uptr AmdgpuPointerInfoMapBase(uptr ptr,
                                     const hsa_amd_pointer_info_t& info) {
  switch (info.type) {
    case HSA_EXT_POINTER_TYPE_RESERVED_ADDR: {
      uptr map_beg = reinterpret_cast<uptr>(info.hostBaseAddress);
      // ROCr sets hostBaseAddress to the OS/KFD reservation base (os_addr).
      // With HSA_AMD_VMEM_ADDRESS_NO_REGISTER the user VA from reserve_align
      // is AlignUp(os_addr, alignment) and is the handle for vmem_address_free;
      // it can differ from hostBaseAddress when alignment > page size.
      if (map_beg && ptr >= map_beg && ptr < map_beg + info.sizeInBytes)
        return map_beg;
      return ptr;
    }
    case HSA_EXT_POINTER_TYPE_HSA:
    case HSA_EXT_POINTER_TYPE_HSA_VMEM:
    case HSA_EXT_POINTER_TYPE_IPC:
      // Match ROCr PtrInfo block_info: prefer host mapping when present.
      if (info.hostBaseAddress)
        return reinterpret_cast<uptr>(info.hostBaseAddress);
      return reinterpret_cast<uptr>(info.agentBaseAddress);
    case HSA_EXT_POINTER_TYPE_LOCKED:
      return reinterpret_cast<uptr>(info.hostBaseAddress);
    default:
      return 0;
  }
}

bool AmdgpuDeviceAllocator::GetPointerInfo(uptr ptr,
                                           DevicePointerInfo* ptr_info) {
  if (!ptr_info)
    return false;

  // GetPointerInfo returns false after AMDGPU runtime shutdown
  if (UNLIKELY(IsRuntimeShutdown())) {
    VReport(1,
            "Amdgpu GetPointerInfo: Runtime shutdown, skipping query for %p\n",
            reinterpret_cast<void*>(ptr));
    return false;
  }

  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);
  hsa_status_t status =
      hsa_amd.pointer_info(reinterpret_cast<void*>(ptr), &info, 0, 0, 0);

  if (status != HSA_STATUS_SUCCESS)
    return false;

  if (info.type == HSA_EXT_POINTER_TYPE_UNKNOWN || !info.sizeInBytes)
    return false;

  const uptr map_beg = AmdgpuPointerInfoMapBase(ptr, info);
  if (!map_beg)
    return false;

  ptr_info->map_beg = map_beg;
  ptr_info->map_size = info.sizeInBytes;
  ptr_info->type = reinterpret_cast<hsa_amd_pointer_type_t>(info.type);

  return true;
}
// Register shutdown system event handler only once
// TODO: Register multiple event handlers if needed in future
void AmdgpuDeviceAllocator::RegisterSystemEventHandlers() {
  uint8_t registered = 0;
  // Check if shutdown event handler is already registered
  if (atomic_compare_exchange_strong(&amdgpu_event_registered, &registered, 1,
                                     memory_order_acq_rel)) {
    // Callback to detect and notify AMDGPU runtime shutdown
    hsa_amd_system_event_callback_t callback = [](const hsa_amd_event_t* event,
                                                  void* data) {
      if (!event)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      if (event->event_type == HSA_AMD_SYSTEM_SHUTDOWN_EVENT)
        AmdgpuDeviceAllocator::NotifyRuntimeShutdown();
      return HSA_STATUS_SUCCESS;
    };
    // Register the event callback
    hsa_status_t status =
        hsa_amd.register_system_event_handler(callback, nullptr);
    // Check as registered if successful
    if (status == HSA_STATUS_SUCCESS)
      VReport(
          1,
          "Amdgpu RegisterSystemEventHandlers: Registered shutdown event \n");
    else {
      VReport(1,
              "Amdgpu RegisterSystemEventHandlers: Failed to register shutdown "
              "event \n");
      atomic_store(&amdgpu_event_registered, 0, memory_order_release);
    }
  }
}

uptr AmdgpuDeviceAllocator::GetPageSize() { return kPageSize_; }

// File-scope singleton: function-local static would emit __cxa_guard_* calls
// that are unavailable when this TU is linked into standalone runtimes.
static VmemGpuReserveTracker vmem_gpu_reserve_tracker;

VmemGpuReserveTracker& VmemGpuReserveTracker::Get() {
  return vmem_gpu_reserve_tracker;
}

void VmemGpuReserveTracker::EnsureInited() {
  if (atomic_load(&inited_, memory_order_acquire))
    return;
  SpinMutexLock l(&mu_);
  if (atomic_load(&inited_, memory_order_relaxed))
    return;
  reservations_.Initialize(0);
  atomic_store(&inited_, 1, memory_order_release);
}

void VmemGpuReserveTracker::OnReserve(uptr ptr, uptr size) {
  EnsureInited();
  SpinMutexLock l(&mu_);
  VmemGpuReservation entry;
  entry.ptr = ptr;
  entry.size = size;
  entry.freed = false;
  reservations_.push_back(entry);
}

VmemGpuReserveTracker::FreeResult VmemGpuReserveTracker::CheckFree(uptr ptr,
                                                                   uptr size) {
  EnsureInited();
  SpinMutexLock l(&mu_);
  for (uptr i = 0; i < reservations_.size(); ++i) {
    const VmemGpuReservation& r = reservations_[i];
    if (r.ptr != ptr)
      continue;
    if (r.size != size)
      return kSizeMismatch;
    if (r.freed)
      return kDoubleFree;
    return kFirstFree;
  }
  return kNotTracked;
}

void VmemGpuReserveTracker::MarkFreed(uptr ptr, uptr size) {
  EnsureInited();
  SpinMutexLock l(&mu_);
  for (uptr i = 0; i < reservations_.size(); ++i) {
    VmemGpuReservation& r = reservations_[i];
    if (r.ptr == ptr && r.size == size) {
      r.freed = true;
      return;
    }
  }
}

}  // namespace __sanitizer
#endif  // SANITIZER_AMDHSA
