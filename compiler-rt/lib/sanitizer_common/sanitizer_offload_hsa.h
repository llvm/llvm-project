//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal HSA declarations without a ROCm dependency.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_OFFLOAD_HSA_H
#define SANITIZER_OFFLOAD_HSA_H

#include <stddef.h>
#include <stdint.h>

#define SANITIZER_HSA_LIBRARY "libhsa-runtime64"

#define SANITIZER_HSA_FUNCTIONS(X)        \
  X(hsa_iterate_agents)                   \
  X(hsa_agent_get_info)                   \
  X(hsa_executable_get_symbol_by_name)    \
  X(hsa_executable_symbol_get_info)       \
  X(hsa_memory_copy)                      \
  X(hsa_amd_memory_pool_allocate)         \
  X(hsa_amd_memory_pool_free)             \
  X(hsa_amd_agents_allow_access)          \
  X(hsa_amd_memory_pool_get_info)         \
  X(hsa_amd_agent_iterate_memory_pools)   \
  X(hsa_system_get_major_extension_table) \
  X(hsa_amd_signal_create)                \
  X(hsa_signal_destroy)                   \
  X(hsa_signal_store_screlease)           \
  X(hsa_signal_wait_scacquire)

extern "C" {

typedef struct hsa_agent_s {
  uint64_t handle;
} hsa_agent_t;
typedef struct hsa_executable_s {
  uint64_t handle;
} hsa_executable_t;
typedef struct hsa_executable_symbol_s {
  uint64_t handle;
} hsa_executable_symbol_t;
typedef struct hsa_amd_memory_pool_s {
  uint64_t handle;
} hsa_amd_memory_pool_t;
typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;
typedef struct hsa_loaded_code_object_s {
  uint64_t handle;
} hsa_loaded_code_object_t;
typedef struct hsa_code_object_reader_s {
  uint64_t handle;
} hsa_code_object_reader_t;

// POSIX file descriptor, or a Win32 HANDLE stored as void* so windows.h stays
// out of this mirror.
#if defined(_WIN32) || defined(_WIN64)
typedef void* hsa_file_t;
#else
typedef int hsa_file_t;
#endif

typedef int64_t hsa_signal_value_t;

typedef enum {
  HSA_STATUS_SUCCESS = 0,
  HSA_STATUS_INFO_BREAK = 0x1,
  HSA_STATUS_ERROR = 0x1000,
} hsa_status_t;

typedef enum {
  HSA_DEVICE_TYPE_CPU = 0,
  HSA_DEVICE_TYPE_GPU = 1,
} hsa_device_type_t;

typedef enum {
  HSA_AGENT_INFO_NAME = 0,
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  HSA_AGENT_INFO_DEVICE = 17,
} hsa_agent_info_t;

typedef enum { HSA_AMD_SEGMENT_GLOBAL = 0 } hsa_amd_segment_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0,
  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5,
} hsa_amd_memory_pool_info_t;

typedef enum {
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2,
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4,
} hsa_amd_memory_pool_global_flag_t;

typedef enum {
  HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 0xA002,
  HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = 0xA00A,
} hsa_amd_agent_info_t;

typedef enum {
  HSA_SIGNAL_CONDITION_NE = 1,
} hsa_signal_condition_t;

typedef enum {
  HSA_WAIT_STATE_BLOCKED = 0,
} hsa_wait_state_t;

typedef enum {
  HSA_SYMBOL_KIND_VARIABLE = 0,
} hsa_symbol_kind_t;

typedef enum {
  HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21,
} hsa_executable_symbol_info_t;

typedef enum { HSA_EXTENSION_AMD_LOADER = 0x201 } hsa_extension_t;

typedef enum {
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND = 2,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT = 3,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE = 4,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE =
      5,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE =
      6,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE = 9,
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE = 10,
} hsa_ven_amd_loader_loaded_code_object_info_t;

typedef enum {
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT = 2,
} hsa_ven_amd_loader_loaded_code_object_kind_t;

typedef enum {
  HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY = 2,
} hsa_ven_amd_loader_code_object_storage_type_t;

// hsa_ven_amd_loader_segment_descriptor_t. storage type is uint32_t so the
// mirror does not depend on the compiler's enum width.
struct hsa_ven_amd_loader_segment_descriptor_t {
  hsa_agent_t agent;
  hsa_executable_t executable;
  uint32_t code_object_storage_type;
  const void* code_object_storage_base;
  size_t code_object_storage_size;
  size_t code_object_storage_offset;
  const void* segment_base;
  size_t segment_size;
};
static_assert(sizeof(hsa_ven_amd_loader_segment_descriptor_t) == 64,
              "layout drift");

// Version 1.04 of HSA_EXTENSION_AMD_LOADER. Slot order is the extension ABI.
struct LoaderApi {
  hsa_status_t (*QueryHostAddress)(const void*, const void**);
  hsa_status_t (*QuerySegmentDescriptors)(
      hsa_ven_amd_loader_segment_descriptor_t*, size_t*);
  hsa_status_t (*QueryExecutable)(const void*, hsa_executable_t*);
  hsa_status_t (*IterateLoadedCodeObjects)(
      hsa_executable_t,
      hsa_status_t (*)(hsa_executable_t, hsa_loaded_code_object_t, void*),
      void*);
  hsa_status_t (*GetCodeObjectInfo)(
      hsa_loaded_code_object_t, hsa_ven_amd_loader_loaded_code_object_info_t,
      void*);
  hsa_status_t (*CreateCodeObjectReaderFromFile)(hsa_file_t, size_t, size_t,
                                                 hsa_code_object_reader_t*);
  hsa_status_t (*IterateExecutables)(hsa_status_t (*)(hsa_executable_t, void*),
                                     void*);
  hsa_status_t (*CodeObjectReaderGetLoadSize)(hsa_code_object_reader_t,
                                              size_t*);
  hsa_status_t (*LoadAgentCodeObjectAtAddress)(hsa_executable_t, hsa_agent_t,
                                               hsa_code_object_reader_t,
                                               const char*, void*, size_t,
                                               hsa_loaded_code_object_t*);
};
static_assert(sizeof(LoaderApi) == 9 * sizeof(void*), "layout drift");

hsa_status_t hsa_init(void);
hsa_status_t hsa_shut_down(void);
hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void*),
                                void* data);
hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                void* value);
hsa_status_t hsa_executable_destroy(hsa_executable_t executable);
hsa_status_t hsa_executable_freeze(hsa_executable_t executable,
                                   const char* options);
hsa_status_t hsa_executable_get_symbol_by_name(hsa_executable_t executable,
                                               const char* symbol_name,
                                               const hsa_agent_t* agent,
                                               hsa_executable_symbol_t* symbol);
hsa_status_t hsa_executable_symbol_get_info(
    hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute,
    void* value);
hsa_status_t hsa_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t,
                             hsa_executable_symbol_t, void*),
    void* data);
hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool,
                                          size_t size, uint32_t flags,
                                          void** ptr);
hsa_status_t hsa_amd_memory_pool_free(void* ptr);
hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                          hsa_amd_memory_pool_info_t attribute,
                                          void* value);
hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void*),
    void* data);
hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents,
                                         const hsa_agent_t* agents,
                                         const uint32_t* flags,
                                         const void* ptr);
hsa_status_t hsa_system_get_major_extension_table(uint16_t extension,
                                                  uint16_t version_major,
                                                  size_t table_length,
                                                  void* table);
hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size);
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value,
                                   uint32_t num_consumers,
                                   const hsa_agent_t* consumers,
                                   uint64_t attributes, hsa_signal_t* signal);
hsa_status_t hsa_signal_destroy(hsa_signal_t signal);
void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal,
                                             hsa_signal_condition_t condition,
                                             hsa_signal_value_t compare_value,
                                             uint64_t timeout_hint,
                                             hsa_wait_state_t wait_state_hint);

}  // extern "C"

#endif  // SANITIZER_OFFLOAD_HSA_H
