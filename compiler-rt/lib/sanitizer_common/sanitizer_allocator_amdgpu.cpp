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

namespace __sanitizer {
struct HsaMemoryFunctions {
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
  hsa_status_t (*vmem_address_free)(void* ptr, size_t size);
};

static HsaMemoryFunctions hsa_amd;

// Always align to page boundary to match current ROCr behavior
static const size_t kPageSize_ = 4096;

bool AmdgpuMemFuncs::Init() {
  hsa_amd.memory_pool_allocate =
      (decltype(hsa_amd.memory_pool_allocate))dlsym(
          RTLD_NEXT, "hsa_amd_memory_pool_allocate");
  hsa_amd.memory_pool_free = (decltype(hsa_amd.memory_pool_free))dlsym(
      RTLD_NEXT, "hsa_amd_memory_pool_free");
  hsa_amd.pointer_info = (decltype(hsa_amd.pointer_info))dlsym(
      RTLD_NEXT, "hsa_amd_pointer_info");
  hsa_amd.vmem_address_reserve_align =
      (decltype(hsa_amd.vmem_address_reserve_align))dlsym(
          RTLD_NEXT, "hsa_amd_vmem_address_reserve_align");
  hsa_amd.vmem_address_free = (decltype(hsa_amd.vmem_address_free))dlsym(
      RTLD_NEXT, "hsa_amd_vmem_address_free");
  if (!hsa_amd.memory_pool_allocate || !hsa_amd.memory_pool_free ||
      !hsa_amd.pointer_info || !hsa_amd.vmem_address_reserve_align ||
      !hsa_amd.vmem_address_free)
    return false;
  return true;
}

void *AmdgpuMemFuncs::Allocate(uptr size, uptr alignment,
                               DeviceAllocationInfo *da_info) {
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

uptr AmdgpuMemFuncs::GetPageSize() { return kPageSize_; }
}  // namespace __sanitizer
#endif  // SANITIZER_AMDGPU
