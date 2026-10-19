//===- InstrProfilingPlatformROCmHSADefs.h - mirrored HSA decls ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal HSA type/enum/function-pointer declarations used by the Linux-only
// supplemental HSA drain (InstrProfilingPlatformROCmHSA.cpp). compiler-rt
// cannot depend on the ROCm headers at build time, and the runtime dlopens
// libhsa-runtime64.so rather than linking it, so the handful of declarations
// the drain needs are mirrored here under a prof_hsa_* prefix.
//
// Values mirror hsa/hsa.h and hsa/hsa_ven_amd_loader.h. These are part of HSA's
// stable, versioned C ABI (libhsa-runtime64.so.1), so they do not shift.
//
//===----------------------------------------------------------------------===//

#ifndef PROFILE_INSTRPROFILINGPLATFORMROCMHSADEFS_H
#define PROFILE_INSTRPROFILINGPLATFORMROCMHSADEFS_H

#include <stddef.h>
#include <stdint.h>

typedef uint32_t prof_hsa_status_t;
#define PROF_HSA_STATUS_SUCCESS ((prof_hsa_status_t)0x0)
#define PROF_HSA_STATUS_INFO_BREAK ((prof_hsa_status_t)0x1)

typedef struct {
  uint64_t handle;
} prof_hsa_agent_t;
typedef struct {
  uint64_t handle;
} prof_hsa_executable_t;
typedef struct {
  uint64_t handle;
} prof_hsa_executable_symbol_t;

typedef uint32_t prof_hsa_agent_info_t;
#define PROF_HSA_AGENT_INFO_NAME ((prof_hsa_agent_info_t)0)
#define PROF_HSA_AGENT_INFO_DEVICE ((prof_hsa_agent_info_t)17)

typedef uint32_t prof_hsa_device_type_t;
#define PROF_HSA_DEVICE_TYPE_GPU ((prof_hsa_device_type_t)1)

typedef uint32_t prof_hsa_symbol_kind_t;
#define PROF_HSA_SYMBOL_KIND_VARIABLE ((prof_hsa_symbol_kind_t)0)

typedef uint32_t prof_hsa_executable_symbol_info_t;
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_TYPE                                   \
  ((prof_hsa_executable_symbol_info_t)0)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH                            \
  ((prof_hsa_executable_symbol_info_t)1)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_NAME                                   \
  ((prof_hsa_executable_symbol_info_t)2)
#define PROF_HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS                       \
  ((prof_hsa_executable_symbol_info_t)21)

#define PROF_HSA_EXTENSION_AMD_LOADER ((uint16_t)0x201)

typedef uint32_t prof_hsa_loader_storage_type_t;

typedef struct {
  prof_hsa_agent_t agent;
  prof_hsa_executable_t executable;
  prof_hsa_loader_storage_type_t code_object_storage_type;
  const void *code_object_storage_base;
  size_t code_object_storage_size;
  size_t code_object_storage_offset;
  const void *segment_base;
  size_t segment_size;
} prof_hsa_loader_segment_descriptor_t;

typedef prof_hsa_status_t (*hsa_init_ty)(void);
typedef prof_hsa_status_t (*hsa_iterate_agents_ty)(
    prof_hsa_status_t (*)(prof_hsa_agent_t, void *), void *);
typedef prof_hsa_status_t (*hsa_agent_get_info_ty)(prof_hsa_agent_t,
                                                   prof_hsa_agent_info_t,
                                                   void *);
typedef prof_hsa_status_t (*hsa_executable_iterate_agent_symbols_ty)(
    prof_hsa_executable_t, prof_hsa_agent_t,
    prof_hsa_status_t (*)(prof_hsa_executable_t, prof_hsa_agent_t,
                          prof_hsa_executable_symbol_t, void *),
    void *);
typedef prof_hsa_status_t (*hsa_executable_symbol_get_info_ty)(
    prof_hsa_executable_symbol_t, prof_hsa_executable_symbol_info_t, void *);
typedef prof_hsa_status_t (*hsa_system_get_major_extension_table_ty)(uint16_t,
                                                                     uint16_t,
                                                                     size_t,
                                                                     void *);
typedef prof_hsa_status_t (*hsa_loader_query_segment_descriptors_ty)(
    prof_hsa_loader_segment_descriptor_t *, size_t *);

/* First two members of hsa_ven_amd_loader_1_00_pfn_t; query_host_address only
 * pads the offset to query_segment_descriptors. */
typedef struct {
  void *query_host_address;
  hsa_loader_query_segment_descriptors_ty query_segment_descriptors;
} prof_hsa_loader_pfn_t;

#endif // PROFILE_INSTRPROFILINGPLATFORMROCMHSADEFS_H
