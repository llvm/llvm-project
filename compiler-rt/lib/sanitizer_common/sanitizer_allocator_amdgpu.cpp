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
#if SANITIZER_AMDGPU
#  include <dlfcn.h>  // For dlsym
#  include "sanitizer_allocator.h"
#  include "sanitizer_atomic.h"

namespace __sanitizer {
struct HsaFunctions {
  // -------------- Memory Allocate/Deallocate Functions ----------------
  hsa_status_t (*memory_pool_allocate)(hsa_amd_memory_pool_t memory_pool,
                                       size_t size, uint32_t flags, void **ptr);
  hsa_status_t (*memory_pool_free)(void *ptr);
  hsa_status_t (*pointer_info)(void *ptr, hsa_amd_pointer_info_t *info,
                               void *(*alloc)(size_t),
                               uint32_t *num_agents_accessible,
                               hsa_agent_t **accessible);
  hsa_status_t (*vmem_address_reserve_align)(void** ptr, size_t size,
                                             uint64_t address,
                                             uint64_t alignment,
                                             uint64_t flags);
  hsa_status_t (*vmem_address_free)(void *ptr, size_t size);

  // ----------------- System Event Register Function -------------------
  hsa_status_t (*register_system_event_handler)(
      hsa_amd_system_event_callback_t callback, void *data);
};

static HsaFunctions hsa_amd;

// Always align to page boundary to match current ROCr behavior
static const size_t kPageSize_ = 4096;

static atomic_uint8_t amdgpu_runtime_shutdown{0};
static atomic_uint8_t amdgpu_event_registered{0};

#  define LOAD_HSA_FUNC_WITH_ERROR_CHECK(func, name, success)         \
    func = (decltype(func))dlsym(RTLD_NEXT, name);                    \
    if (!func) {                                                      \
      VReport(2, "Amdgpu Init: Failed to load " #name " function\n"); \
      success = false;                                                \
    }

// Check AMDGPU runtime shutdown state
bool AmdgpuMemFuncs::IsAmdgpuRuntimeShutdown() {
  return static_cast<bool>(
      atomic_load(&amdgpu_runtime_shutdown, memory_order_acquire));
}

// Notify AMDGPU runtime shutdown to allocator
void AmdgpuMemFuncs::NotifyAmdgpuRuntimeShutdown() {
  uint8_t shutdown = 0;
  if (atomic_compare_exchange_strong(&amdgpu_runtime_shutdown, &shutdown, 1,
                                     memory_order_acq_rel)) {
    VReport(2, "Amdgpu Allocator: AMDGPU runtime shutdown detected\n");
  }
}

bool AmdgpuMemFuncs::Init() {
  bool success = true;
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.memory_pool_allocate,
                                 "hsa_amd_memory_pool_allocate", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.memory_pool_free,
                                 "hsa_amd_memory_pool_free", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.pointer_info, "hsa_amd_pointer_info",
                                 success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.vmem_address_reserve_align,
                                 "hsa_amd_vmem_address_reserve_align", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.vmem_address_free,
                                 "hsa_amd_vmem_address_free", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.register_system_event_handler,
                                 "hsa_amd_register_system_event_handler",
                                 success);
  if (!success) {
    VReport(1, "Amdgpu Init: Failed to load AMDGPU runtime functions\n");
    return false;
  }
  return true;
}

void *AmdgpuMemFuncs::Allocate(uptr size, uptr alignment,
                               DeviceAllocationInfo *da_info) {
  // Do not allocate if AMDGPU runtime is shutdown
  if (UNLIKELY(IsAmdgpuRuntimeShutdown())) {
    VReport(1,
            "Amdgpu Allocate: Runtime shutdown, skipping allocation for size "
            "%zu alignment %zu\n",
            size, alignment);
    return nullptr;
  }

  AmdgpuAllocationInfo *aa_info =
      reinterpret_cast<AmdgpuAllocationInfo *>(da_info);
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

void AmdgpuMemFuncs::Deallocate(void *p) {
  // Deallocate does nothing after AMDGPU runtime shutdown
  if (UNLIKELY(IsAmdgpuRuntimeShutdown())) {
    VReport(
        1,
        "Amdgpu Deallocate: Runtime shutdown, skipping deallocation for %p\n",
        reinterpret_cast<void*>(p));
    return;
  }

  DevicePointerInfo DevPtrInfo;
  if (AmdgpuMemFuncs::GetPointerInfo(reinterpret_cast<uptr>(p), &DevPtrInfo)) {
    if (DevPtrInfo.type == HSA_EXT_POINTER_TYPE_HSA) {
      UNUSED hsa_status_t status = hsa_amd.memory_pool_free(p);
    } else if (DevPtrInfo.type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR) {
      UNUSED hsa_status_t status =
          hsa_amd.vmem_address_free(p, DevPtrInfo.map_size);
    }
  }
}

bool AmdgpuMemFuncs::GetPointerInfo(uptr ptr, DevicePointerInfo* ptr_info) {
  // GetPointerInfo returns false after AMDGPU runtime shutdown
  if (UNLIKELY(IsAmdgpuRuntimeShutdown())) {
    VReport(1,
            "Amdgpu GetPointerInfo: Runtime shutdown, skipping query for %p\n",
            reinterpret_cast<void*>(ptr));
    return false;
  }

  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);
  hsa_status_t status =
    hsa_amd.pointer_info(reinterpret_cast<void *>(ptr), &info, 0, 0, 0);

  if (status != HSA_STATUS_SUCCESS)
    return false;

  if (info.type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR)
    ptr_info->map_beg = reinterpret_cast<uptr>(info.hostBaseAddress);
  else if (info.type == HSA_EXT_POINTER_TYPE_HSA)
    ptr_info->map_beg = reinterpret_cast<uptr>(info.agentBaseAddress);
  ptr_info->map_size = info.sizeInBytes;
  ptr_info->type = reinterpret_cast<hsa_amd_pointer_type_t>(info.type);

  return true;
}
 // Register shutdown system event handler only once
 // TODO: Register multiple event handlers if needed in future
void AmdgpuMemFuncs::RegisterSystemEventHandlers() {
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
        AmdgpuMemFuncs::NotifyAmdgpuRuntimeShutdown();
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

uptr AmdgpuMemFuncs::GetPageSize() { return kPageSize_; }
}  // namespace __sanitizer
#endif  // SANITIZER_AMDGPU
