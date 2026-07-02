//===-- sanitizer_allocator_amdgpu.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGPU / HSA device memory backend for DeviceAllocatorT.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_H
#  error This file must be included after sanitizer_allocator.h
#endif

#if SANITIZER_AMDHSA

static const DeviceAllocationType DAT_AMDGPU =
    static_cast<DeviceAllocationType>(1);

class AmdgpuDeviceAllocator {
 public:
  static constexpr bool kEnableDeviceBackend = true;

  static bool Init(bool allow_dlopen = false);
  static void* Allocate(uptr size, uptr alignment,
                        DeviceAllocationInfo* da_info);
  static void Deallocate(void* p);
  static bool GetPointerInfo(uptr ptr, DevicePointerInfo* ptr_info);
  static uptr GetPageSize();
  static void RegisterSystemEventHandlers();
  static bool IsRuntimeShutdown();
  static void ClearRuntimeShutdownState();
  static void NoteDeviceAllocatorFailure(DeviceAllocationInfo* da_info,
                                         DeviceAllocFailure failure);

 private:
  static void NotifyRuntimeShutdown();
};

// GPU-only hsa_amd_vmem_address_reserve_align (without
// HSA_AMD_VMEM_ADDRESS_NO_REGISTER) returns KFD "address only" VA that is not
// host-writable. Sanitizer runtimes that cannot place metadata in the reserved
// range track such reservations here for double-free detection on free.
class VmemGpuReserveTracker {
 public:
  enum FreeResult {
    kNotTracked,
    kFirstFree,
    kSizeMismatch,
    kDoubleFree,
  };

  static VmemGpuReserveTracker& Get();

  void OnReserve(uptr ptr, uptr size);
  // CheckFree validates without updating state; call MarkFreed only after the
  // real hsa_amd_vmem_address_free succeeds.
  FreeResult CheckFree(uptr ptr, uptr size);
  void MarkFreed(uptr ptr, uptr size);

 private:
  struct VmemGpuReservation {
    uptr ptr;
    uptr size;
    bool freed;
  };

  void EnsureInited();

  StaticSpinMutex mu_;
  InternalMmapVectorNoCtor<VmemGpuReservation> reservations_;
  atomic_uint8_t inited_;
};

// True when ROCr reserves host-accessible VA (e.g. HIP managed / SVM).
inline bool AmdgpuVmemReserveUsesHostMapping(u64 flags) {
  return (flags & HSA_AMD_VMEM_ADDRESS_NO_REGISTER) != 0;
}

struct AmdgpuAllocationInfo : public DeviceAllocationInfo {
  AmdgpuAllocationInfo() : DeviceAllocationInfo(DAT_AMDGPU) {
    status = HSA_STATUS_SUCCESS;
    alloc_func = nullptr;
  }
  // If allocation fails without an HSA API status, record one so callers never
  // see *ptr == nullptr with status still SUCCESS.
  void EnsureFailureStatus(hsa_status_t err) {
    if (status == HSA_STATUS_SUCCESS)
      status = err;
  }

  hsa_status_t status;
  void* alloc_func;
  hsa_amd_memory_pool_t memory_pool;
  u64 alignment;
  u64 address;
  u64 flags64;
  usize size;
  u32 flags;
  void* ptr;
};

// AMDGPU device heap for CombinedAllocator's third template parameter:
// DeviceAllocatorT<Primary, AmdgpuDeviceAllocator>.
template <class PrimaryAllocator>
using AmdgpuDeviceAllocatorT =
    DeviceAllocatorT<PrimaryAllocator, AmdgpuDeviceAllocator>;

#endif  // SANITIZER_AMDHSA
