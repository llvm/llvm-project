//===-- asan_hsa_allocator.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGPU HSA allocation wrappers for AddressSanitizer (SANITIZER_AMDHSA).
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_AMDHSA

#  include "asan_allocator.h"
#  include "asan_hsa_allocator.h"
#  include "asan_interceptors.h"
#  include "asan_internal.h"
#  include "asan_mapping.h"
#  include "asan_poisoning.h"
#  include "asan_report.h"
#  include "sanitizer_common/sanitizer_allocator_checks.h"
#  include "sanitizer_common/sanitizer_errno.h"
#  include "sanitizer_common/sanitizer_hsa.h"

namespace __asan {

DECLARE_REAL(hsa_status_t, hsa_init);
DECLARE_REAL(hsa_status_t, hsa_amd_agents_allow_access, uint32_t num_agents,
             const hsa_agent_t* agents, const uint32_t* flags, const void* ptr)
DECLARE_REAL(hsa_status_t, hsa_amd_memory_pool_allocate,
             hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags,
             void** ptr)
DECLARE_REAL(hsa_status_t, hsa_amd_memory_pool_free, void* ptr)
DECLARE_REAL(hsa_status_t, hsa_amd_ipc_memory_create, void* ptr, size_t len,
             hsa_amd_ipc_memory_t* handle)
DECLARE_REAL(hsa_status_t, hsa_amd_ipc_memory_attach,
             const hsa_amd_ipc_memory_t* handle, size_t len,
             uint32_t num_agents, const hsa_agent_t* mapping_agents,
             void** mapped_ptr)
DECLARE_REAL(hsa_status_t, hsa_amd_ipc_memory_detach, void* mapped_ptr)
DECLARE_REAL(hsa_status_t, hsa_amd_vmem_address_reserve_align, void** ptr,
             size_t size, uint64_t address, uint64_t alignment, uint64_t flags)
DECLARE_REAL(hsa_status_t, hsa_amd_vmem_address_free, void* ptr, size_t size)
DECLARE_REAL(hsa_status_t, hsa_amd_pointer_info, const void* ptr,
             hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
             uint32_t* num_agents_accessible, hsa_agent_t** accessible)

// Always align to page boundary to match current ROCr behavior.
static const size_t kPageSize_ = 4096;

#  if SANITIZER_CAN_USE_ALLOCATOR64
static_assert(AP64<LocalAddressSpaceView>::kMetadataSize == 0,
              "HSA IPC/VMem wrappers require zero allocator metadata");
#  else
static_assert(AP32<LocalAddressSpaceView>::kMetadataSize == 0,
              "HSA IPC/VMem wrappers require zero allocator metadata");
#  endif

hsa_status_t asan_hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr,
    BufferedStackTrace* stack) {
  AmdgpuAllocationInfo aa_info;
  aa_info.alloc_func =
      reinterpret_cast<void*>((uptr)&__asan::asan_hsa_amd_memory_pool_allocate);
  aa_info.memory_pool = memory_pool;
  aa_info.size = size;
  aa_info.flags = flags;
  aa_info.ptr = nullptr;
  SetErrnoOnNull(*ptr = AsanHsaAllocate(size, kPageSize_, stack, &aa_info));
  return aa_info.status;
}

hsa_status_t asan_hsa_amd_memory_pool_free(void* ptr,
                                           BufferedStackTrace* stack) {
  void* p = AsanHsaGetBlockBegin(ptr);
  if (p) {
    AsanHsaDeallocate(ptr, stack);
    return HSA_STATUS_SUCCESS;
  }
  return REAL(hsa_amd_memory_pool_free)(ptr);
}

hsa_status_t asan_hsa_amd_agents_allow_access(uint32_t num_agents,
                                              const hsa_agent_t* agents,
                                              const uint32_t* flags,
                                              const void* ptr,
                                              BufferedStackTrace* stack) {
  (void)stack;
  void* p = AsanHsaGetBlockBegin(ptr);
  return REAL(hsa_amd_agents_allow_access)(num_agents, agents, flags,
                                           p ? p : ptr);
}

hsa_status_t asan_hsa_amd_ipc_memory_create(void* ptr, size_t len,
                                            hsa_amd_ipc_memory_t* handle) {
  void* ptr_;
  size_t len_;
  if (AsanHsaTranslateIpcCreate(ptr, len, &ptr_, &len_))
    return REAL(hsa_amd_ipc_memory_create)(ptr_, len_, handle);
  return REAL(hsa_amd_ipc_memory_create)(ptr, len, handle);
}

hsa_status_t asan_hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* handle,
                                            size_t len, uint32_t num_agents,
                                            const hsa_agent_t* mapping_agents,
                                            void** mapped_ptr) {
  size_t len_ = len + kPageSize_;
  hsa_status_t status = REAL(hsa_amd_ipc_memory_attach)(
      handle, len_, num_agents, mapping_agents, mapped_ptr);
  if (status == HSA_STATUS_SUCCESS && mapped_ptr) {
    uptr mapped_base = reinterpret_cast<uptr>(*mapped_ptr);
    uptr user_beg = mapped_base + kPageSize_;
    uptr tail_beg = RoundUpTo(user_beg + len, ASAN_SHADOW_GRANULARITY);
    uptr mapped_end = mapped_base + kPageSize_ + RoundUpTo(len, kPageSize_);

    PoisonShadow(mapped_base, kPageSize_, kAsanHeapLeftRedzoneMagic);

    if (mapped_end > tail_beg)
      PoisonShadow(tail_beg, mapped_end - tail_beg, kAsanHeapLeftRedzoneMagic);

    uptr size_rounded_down = RoundDownTo(len, ASAN_SHADOW_GRANULARITY);
    if (size_rounded_down)
      PoisonShadow(user_beg, size_rounded_down, 0);

    if (len != size_rounded_down && CanPoisonMemory()) {
      u8* shadow = (u8*)MemToShadow(user_beg + size_rounded_down);
      *shadow = flags()->poison_partial
                    ? static_cast<u8>(len & (ASAN_SHADOW_GRANULARITY - 1))
                    : 0;
    }

    *mapped_ptr = reinterpret_cast<void*>(user_beg);
  }
  return status;
}

hsa_status_t asan_hsa_amd_ipc_memory_detach(void* mapped_ptr) {
  uptr mapped_base = reinterpret_cast<uptr>(mapped_ptr) - kPageSize_;

  // Snapshot mapping size before detach: ROCr may return an error while the
  // import is still live (e.g. thunk_bo path if hsaKmtMemoryVaUnmap fails).
  // Only clear ASan shadow after a successful detach so redzones stay valid
  // if teardown did not complete.
  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);
  uptr mapped_sz = 0;
  if (REAL(hsa_amd_pointer_info)(reinterpret_cast<void*>(mapped_base), &info,
                                 nullptr, nullptr,
                                 nullptr) == HSA_STATUS_SUCCESS)
    mapped_sz = static_cast<uptr>(info.sizeInBytes);

  const hsa_status_t status =
      REAL(hsa_amd_ipc_memory_detach)(reinterpret_cast<void*>(mapped_base));
  if (status == HSA_STATUS_SUCCESS && mapped_sz) {
    PoisonShadow(mapped_base, mapped_sz, 0);
    FlushUnneededASanShadowMemory(mapped_base, mapped_sz);
  }
  return status;
}

hsa_status_t asan_hsa_amd_vmem_address_reserve_align(
    void** ptr, size_t size, uint64_t address, uint64_t alignment,
    uint64_t flags, BufferedStackTrace* stack) {
  // Bypass the tracking for a fixed address since it cannot be supported.
  // Reasons:
  //  1. Address may not meet the alignment/page-size requirement.
  //  2. Requested range overlaps an existing reserved/mapped range.
  //  3. Insufficient VA space to honor that exact placement.
  if (address)
    return REAL(hsa_amd_vmem_address_reserve_align)(ptr, size, address,
                                                    alignment, flags);

  if (alignment < kPageSize_)
    alignment = kPageSize_;

  if (UNLIKELY(!IsPowerOfTwo(alignment))) {
    errno = errno_EINVAL;
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (!__sanitizer::AmdgpuVmemReserveUsesHostMapping(flags)) {
    const hsa_status_t status = REAL(hsa_amd_vmem_address_reserve_align)(
        ptr, size, address, alignment, flags);
    if (status == HSA_STATUS_SUCCESS && ptr && *ptr)
      __sanitizer::VmemGpuReserveTracker::Get().OnReserve(
          reinterpret_cast<uptr>(*ptr), size);
    return status;
  }

  AmdgpuAllocationInfo aa_info;
  aa_info.alloc_func = reinterpret_cast<void*>(
      (uptr)&__asan::asan_hsa_amd_vmem_address_reserve_align);
  aa_info.memory_pool = {0};
  aa_info.size = size;
  aa_info.flags64 = flags;
  aa_info.address = 0;
  aa_info.alignment = alignment;
  aa_info.ptr = nullptr;
  SetErrnoOnNull(*ptr = AsanHsaAllocate(size, alignment, stack, &aa_info));

  return aa_info.status;
}

hsa_status_t asan_hsa_amd_vmem_address_free(void* ptr, size_t size,
                                            BufferedStackTrace* stack) {
  if (UNLIKELY(!IsAligned(reinterpret_cast<uptr>(ptr), kPageSize_))) {
    errno = errno_EINVAL;
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (size == 0) {
    errno = errno_EINVAL;
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (AsanHsaGetBlockBegin(ptr)) {
    if (!AsanHsaIsVmemFreeValid(ptr, size)) {
      errno = errno_EINVAL;
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    AsanHsaDeallocate(ptr, stack);
    return HSA_STATUS_SUCCESS;
  }

  // GPU-only free: only supported when host-accessible VA (NO_REGISTER) is
  // used.
  using VmemGpuReserveTracker = __sanitizer::VmemGpuReserveTracker;
  const uptr ptr_uptr = reinterpret_cast<uptr>(ptr);
  switch (VmemGpuReserveTracker::Get().CheckFree(ptr_uptr, size)) {
    case VmemGpuReserveTracker::kNotTracked:
      break;
    case VmemGpuReserveTracker::kFirstFree: {
      const hsa_status_t status = REAL(hsa_amd_vmem_address_free)(ptr, size);
      if (status == HSA_STATUS_SUCCESS)
        VmemGpuReserveTracker::Get().MarkFreed(ptr_uptr, size);
      return status;
    }
    case VmemGpuReserveTracker::kSizeMismatch:
      errno = errno_EINVAL;
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    case VmemGpuReserveTracker::kDoubleFree:
      ReportDoubleFree(ptr_uptr, stack);
      return HSA_STATUS_SUCCESS;
  }
  // Passthrough: untracked vmem frees (eg: fixed-address reservations) go
  // straight to ROCr.
  return REAL(hsa_amd_vmem_address_free)(ptr, size);
}

hsa_status_t asan_hsa_amd_pointer_info(const void* ptr,
                                       hsa_amd_pointer_info_t* info,
                                       void* (*alloc)(size_t),
                                       uint32_t* num_agents_accessible,
                                       hsa_agent_t** accessible) {
  // Device/pool mappings are keyed by the ROCr reservation base (map_beg), not
  // the user pointer returned from intercepted allocate/reserve.
  void* hsa_map_base;
  uptr used_size;
  uptr offset;
  if (!AsanHsaGetLiveMappingInfo(ptr, &hsa_map_base, &used_size, &offset))
    return REAL(hsa_amd_pointer_info)(ptr, info, alloc, num_agents_accessible,
                                      accessible);

  hsa_status_t status = REAL(hsa_amd_pointer_info)(
      hsa_map_base, info, alloc, num_agents_accessible, accessible);
  if (status == HSA_STATUS_SUCCESS && info) {
    // User VA may be above hsa_map_base when VMem API's(i.e
    // hsa_amd_vmem_address_reserve_align())  uses alignment > page size (device
    // allocator padding and ASan redzones/headers).
    // hostBaseAddress must reflect the user-visible reservation base (ROCr may
    // report os_addr below an aligned user VA for NO_REGISTER reserves).
    if (info->hostBaseAddress)
      info->hostBaseAddress = reinterpret_cast<void*>(
          reinterpret_cast<uptr>(info->hostBaseAddress) + offset);
    // agentBaseAddress is NULL for unmapped RESERVED_ADDR (no GPU backing).
    // Do not turn NULL + offset into a fake agent address.
    if (info->agentBaseAddress)
      info->agentBaseAddress = reinterpret_cast<void*>(
          reinterpret_cast<uptr>(info->agentBaseAddress) + offset);
    info->sizeInBytes = used_size;
  }
  return status;
}

hsa_status_t asan_hsa_init() {
  hsa_status_t status = REAL(hsa_init)();
  if (status == HSA_STATUS_SUCCESS) {
    // Only clear state when recovering from a prior shutdown (avoids clearing
    // amdgpu_event_registered on every refcount bump and re-registering).
    if (__sanitizer::AmdgpuDeviceAllocator::IsRuntimeShutdown())
      __sanitizer::AmdgpuDeviceAllocator::ClearRuntimeShutdownState();
    // Load HSA entry points once the runtime is up; device allocator may stay
    // disabled, but interceptors and RegisterSystemEventHandlers need them.
    if (__sanitizer::AmdgpuDeviceAllocator::Init(/*allow_dlopen=*/true))
      __sanitizer::AmdgpuDeviceAllocator::RegisterSystemEventHandlers();
  }
  return status;
}

}  // namespace __asan

#endif  // SANITIZER_AMDHSA
