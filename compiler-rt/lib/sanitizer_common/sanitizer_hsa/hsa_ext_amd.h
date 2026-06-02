//===-- sanitizer_hsa/hsa_ext_amd.h ----------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal AMD HSA extension declarations used by compiler-rt host sanitizers.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_HSA_HSA_EXT_AMD_H_
#define SANITIZER_HSA_HSA_EXT_AMD_H_

#include "hsa.h"

#ifndef __cplusplus
#  include <stdbool.h>
#endif

#define HSA_AMD_INTERFACE_VERSION_MAJOR 1
#define HSA_AMD_INTERFACE_VERSION_MINOR 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hsa_amd_memory_pool_s {
  uint64_t handle;
} hsa_amd_memory_pool_t;

typedef enum {
  HSA_EXT_POINTER_TYPE_UNKNOWN = 0,
  HSA_EXT_POINTER_TYPE_HSA = 1,
  HSA_EXT_POINTER_TYPE_LOCKED = 2,
  HSA_EXT_POINTER_TYPE_GRAPHICS = 3,
  HSA_EXT_POINTER_TYPE_IPC = 4,
  HSA_EXT_POINTER_TYPE_RESERVED_ADDR = 5,
  HSA_EXT_POINTER_TYPE_HSA_VMEM = 6,
} hsa_amd_pointer_type_t;

typedef struct hsa_amd_pointer_info_s {
  uint32_t size;
  hsa_amd_pointer_type_t type;
  void* agentBaseAddress;
  void* hostBaseAddress;
  size_t sizeInBytes;
} hsa_amd_pointer_info_t;

typedef struct hsa_amd_ipc_memory_s {
  uint32_t handle[8];
} hsa_amd_ipc_memory_t;

typedef enum hsa_amd_sdma_engine_id {
  HSA_AMD_SDMA_ENGINE_0 = 0x1,
  HSA_AMD_SDMA_ENGINE_1 = 0x2,
  HSA_AMD_SDMA_ENGINE_2 = 0x4,
  HSA_AMD_SDMA_ENGINE_3 = 0x8,
  HSA_AMD_SDMA_ENGINE_4 = 0x10,
  HSA_AMD_SDMA_ENGINE_5 = 0x20,
  HSA_AMD_SDMA_ENGINE_6 = 0x40,
  HSA_AMD_SDMA_ENGINE_7 = 0x80,
  HSA_AMD_SDMA_ENGINE_8 = 0x100,
  HSA_AMD_SDMA_ENGINE_9 = 0x200,
  HSA_AMD_SDMA_ENGINE_10 = 0x400,
  HSA_AMD_SDMA_ENGINE_11 = 0x800,
  HSA_AMD_SDMA_ENGINE_12 = 0x1000,
  HSA_AMD_SDMA_ENGINE_13 = 0x2000,
  HSA_AMD_SDMA_ENGINE_14 = 0x4000,
  HSA_AMD_SDMA_ENGINE_15 = 0x8000,
} hsa_amd_sdma_engine_id_t;

typedef enum hsa_amd_vmem_address_reserve_flag_s {
  HSA_AMD_VMEM_ADDRESS_NO_REGISTER = (1UL << 0),
} hsa_amd_vmem_address_reserve_flag_t;

typedef enum hsa_amd_event_type_s {
  HSA_AMD_GPU_MEMORY_FAULT_EVENT = 0,
  HSA_AMD_GPU_MEMORY_ERROR_EVENT = 1,
  HSA_AMD_SYSTEM_SHUTDOWN_EVENT = 2,
} hsa_amd_event_type_t;

typedef struct hsa_amd_gpu_memory_fault_info_s {
  hsa_agent_t agent;
  uint64_t virtual_address;
  uint32_t fault_reason_mask;
} hsa_amd_gpu_memory_fault_info_t;

typedef struct hsa_amd_event_s {
  hsa_amd_event_type_t event_type;
  union {
    hsa_amd_gpu_memory_fault_info_t memory_fault;
  };
} hsa_amd_event_t;

typedef hsa_status_t (*hsa_amd_system_event_callback_t)(
    const hsa_amd_event_t* event, void* data);

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool,
                                          size_t size, uint32_t flags,
                                          void** ptr);
hsa_status_t hsa_amd_memory_pool_free(void* ptr);
hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents,
                                         const hsa_agent_t* agents,
                                         const uint32_t* flags,
                                         const void* ptr);
hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent,
                                       const void* src, hsa_agent_t src_agent,
                                       size_t size, uint32_t num_dep_signals,
                                       const hsa_signal_t* dep_signals,
                                       hsa_signal_t completion_signal);
hsa_status_t hsa_amd_memory_async_copy_on_engine(
    void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal, hsa_amd_sdma_engine_id_t engine_id,
    bool force_copy_on_sdma);
hsa_status_t hsa_amd_ipc_memory_create(void* ptr, size_t len,
                                       hsa_amd_ipc_memory_t* handle);
hsa_status_t hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* handle,
                                       size_t len, uint32_t num_agents,
                                       const hsa_agent_t* mapping_agents,
                                       void** mapped_ptr);
hsa_status_t hsa_amd_ipc_memory_detach(void* mapped_ptr);
hsa_status_t hsa_amd_vmem_address_reserve_align(void** va, size_t size,
                                                uint64_t address,
                                                uint64_t alignment,
                                                uint64_t flags);
hsa_status_t hsa_amd_vmem_address_free(void* va, size_t size);
hsa_status_t hsa_amd_pointer_info(const void* ptr, hsa_amd_pointer_info_t* info,
                                  void* (*alloc)(size_t),
                                  uint32_t* num_agents_accessible,
                                  hsa_agent_t** accessible);
hsa_status_t hsa_amd_register_system_event_handler(
    hsa_amd_system_event_callback_t callback, void* data);

#ifdef __cplusplus
}
#endif

#endif  // SANITIZER_HSA_HSA_EXT_AMD_H_
