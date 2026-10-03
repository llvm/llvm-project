//===-- asan_hsa_allocator.h ---------------------------------- C++ -*-===//
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

#ifndef ASAN_HSA_ALLOCATOR_H
#define ASAN_HSA_ALLOCATOR_H

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_AMDHSA

#  include "asan_stack.h"
#  include "sanitizer_common/sanitizer_hsa.h"

namespace __asan {

hsa_status_t asan_hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr,
    BufferedStackTrace* stack);
hsa_status_t asan_hsa_amd_memory_pool_free(void* ptr,
                                           BufferedStackTrace* stack);
hsa_status_t asan_hsa_amd_agents_allow_access(uint32_t num_agents,
                                              const hsa_agent_t* agents,
                                              const uint32_t* flags,
                                              const void* ptr,
                                              BufferedStackTrace* stack);
hsa_status_t asan_hsa_amd_ipc_memory_create(void* ptr, size_t len,
                                            hsa_amd_ipc_memory_t* handle);
hsa_status_t asan_hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* handle,
                                            size_t len, uint32_t num_agents,
                                            const hsa_agent_t* mapping_agents,
                                            void** mapped_ptr);
hsa_status_t asan_hsa_amd_ipc_memory_detach(void* mapped_ptr);
hsa_status_t asan_hsa_amd_vmem_address_reserve_align(void** ptr, size_t size,
                                                     uint64_t address,
                                                     uint64_t alignment,
                                                     uint64_t flags,
                                                     BufferedStackTrace* stack);
hsa_status_t asan_hsa_amd_vmem_address_free(void* ptr, size_t size,
                                            BufferedStackTrace* stack);
hsa_status_t asan_hsa_amd_pointer_info(const void* ptr,
                                       hsa_amd_pointer_info_t* info,
                                       void* (*alloc)(size_t),
                                       uint32_t* num_agents_accessible,
                                       hsa_agent_t** accessible);
hsa_status_t asan_hsa_init();

}  // namespace __asan

#endif  // SANITIZER_AMDHSA

#endif  // ASAN_HSA_ALLOCATOR_H
