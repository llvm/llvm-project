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
#include <sys/mman.h>

namespace __sanitizer {
struct HsaMemoryFunctions {
  hsa_status_t (*memory_pool_allocate)(hsa_amd_memory_pool_t memory_pool,
                                       size_t size, uint32_t flags, void **ptr);
  hsa_status_t (*memory_pool_free)(void *ptr);
  hsa_status_t (*memory_pool_get_info)(hsa_amd_memory_pool_t memory_pool,
                                       hsa_amd_memory_pool_info_t attribute,
                                       void *value);
  hsa_status_t (*pointer_info)(void *ptr, hsa_amd_pointer_info_t *info,
                               void *(*alloc)(size_t),
                               uint32_t *num_agents_accessible,
                               hsa_agent_t **accessible);
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
  hsa_amd.memory_pool_get_info = (decltype(hsa_amd.memory_pool_get_info))dlsym(
      RTLD_NEXT, "hsa_amd_memory_pool_get_info");
  hsa_amd.pointer_info = (decltype(hsa_amd.pointer_info))dlsym(
      RTLD_NEXT, "hsa_amd_pointer_info");
  if (!hsa_amd.memory_pool_allocate || !hsa_amd.memory_pool_free ||
      !hsa_amd.pointer_info || !hsa_amd.memory_pool_get_info)
    return false;
  else
    return true;
}

void *AmdgpuMemFuncs::Allocate(uptr size, uptr alignment,
                               DeviceAllocationInfo *da_info) {
  AmdgpuAllocationInfo *aa_info =
      reinterpret_cast<AmdgpuAllocationInfo *>(da_info);
  hsa_status_t status;

  status = hsa_amd.memory_pool_allocate(aa_info->memory_pool, size,
                                        aa_info->flags, &aa_info->ptr);
  if (status != HSA_STATUS_SUCCESS)
    goto Fail;

  if (!aa_info->remap_first_device_page)
    return aa_info->ptr;

  // Device memory is mapped as MTRR type WC (write-combined) which will cause
  // failure for atomic_compare_exchange_strong on some platforms. As ASAN
  // logic requires atomic_compare_exchange_strong to work, we will remap the
  // first device memory page to regular host page so that ASAN logic can work
  // as expected. We will remap the original device memory page back when the
  // allocation is freed, because that page might be reused by HSA runtimes.
  //
  hsa_amd_memory_pool_location_t loc;
  status = hsa_amd.memory_pool_get_info(
    aa_info->memory_pool, HSA_AMD_MEMORY_POOL_INFO_LOCATION, &loc);

  if (loc == HSA_AMD_MEMORY_POOL_LOCATION_GPU) {
    void *remapped_device_page, *p;
    // Map a normal page and its vitual address will be used as the target of
    // device memory remap
    remapped_device_page = reinterpret_cast<void *>(
      internal_mmap(nullptr, kPageSize_, PROT_WRITE | PROT_READ,
                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
    if (internal_iserror(reinterpret_cast<uptr>(remapped_device_page))) {
      status = HSA_STATUS_ERROR;
      goto Fail;
    }
    // Remap device memory
    p = reinterpret_cast<void *>(
      internal_mremap(aa_info->ptr, kPageSize_, kPageSize_,
                      MREMAP_FIXED | MREMAP_MAYMOVE, remapped_device_page));
    if (internal_iserror(reinterpret_cast<uptr>(p)) || p == aa_info->ptr) {
      status = HSA_STATUS_ERROR;
      goto Fail;
    }
    // Map a normal page in place of the original device memory
    p = reinterpret_cast<void *>(
      internal_mmap(aa_info->ptr, kPageSize_, PROT_WRITE | PROT_READ,
                    MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED, -1, 0));
    if (internal_iserror(reinterpret_cast<uptr>(p)) || p != aa_info->ptr) {
      status = HSA_STATUS_ERROR;
      goto Fail;
    }
    aa_info->remapped_device_page = remapped_device_page;
  }

  return aa_info->ptr;

Fail:
  if (aa_info->ptr)
    hsa_amd.memory_pool_free(aa_info->ptr);
  aa_info->status = status;
  aa_info->ptr = nullptr;
  return aa_info->ptr;
}

void AmdgpuMemFuncs::Deallocate(void *p, DeviceAllocationInfo *da_info) {
  hsa_status_t status;

  if (da_info && da_info->remapped_device_page) {
    void *p_ = reinterpret_cast<void *>(
      internal_mremap(da_info->remapped_device_page, kPageSize_, kPageSize_,
                      MREMAP_FIXED | MREMAP_MAYMOVE, p));
    if (internal_iserror(reinterpret_cast<uptr>(p_)) || p_ != p)
      status = HSA_STATUS_ERROR;
  }

  status = hsa_amd.memory_pool_free(p);
}

bool AmdgpuMemFuncs::GetPointerInfo(uptr ptr, DevivePointerInfo *ptr_info) {
  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);
  hsa_status_t status =
    hsa_amd.pointer_info(reinterpret_cast<void *>(ptr), &info, 0, 0, 0);

  if (status != HSA_STATUS_SUCCESS)
    return false;

  ptr_info->map_beg = reinterpret_cast<uptr>(info.agentBaseAddress);
  ptr_info->map_size = info.sizeInBytes;

  return true;
}

uptr AmdgpuMemFuncs::GetPageSize() { return kPageSize_; }
}  // namespace __sanitizer
#endif  // SANITIZER_AMDGPU
