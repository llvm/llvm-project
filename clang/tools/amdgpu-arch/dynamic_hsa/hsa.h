//===--- amdgpu/dynamic_hsa/hsa.h --------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The parts of the hsa api that are presently in use by the amdgpu plugin
//
//===----------------------------------------------------------------------===//
#ifndef HSA_RUNTIME_INC_HSA_H_
#define HSA_RUNTIME_INC_HSA_H_

#include <stddef.h>
#include <stdint.h>

// Detect and set large model builds.
#undef HSA_LARGE_MODEL
#if defined(__LP64__) || defined(_M_X64)
#define HSA_LARGE_MODEL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HSA_STATUS_SUCCESS = 0x0,
  HSA_STATUS_INFO_BREAK = 0x1,
  HSA_STATUS_ERROR = 0x1000,
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010,
  HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,
} hsa_status_t;

hsa_status_t hsa_status_string(hsa_status_t status, const char **status_string);

hsa_status_t hsa_init();

hsa_status_t hsa_shut_down();

typedef struct hsa_agent_s {
  uint64_t handle;
} hsa_agent_t;

typedef enum {
  HSA_DEVICE_TYPE_CPU = 0,
  HSA_DEVICE_TYPE_GPU = 1,
  HSA_DEVICE_TYPE_DSP = 2
} hsa_device_type_t;

typedef enum {
  HSA_AGENT_INFO_NAME = 0,
  HSA_AGENT_INFO_PROFILE = 4,
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7,
  HSA_AGENT_INFO_GRID_MAX_DIM = 9,
  HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14,
  HSA_AGENT_INFO_DEVICE = 17,
} hsa_agent_info_t;

hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                void *value);

hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent,
                                                         void *data),
                                void *data);

typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;

#ifdef HSA_LARGE_MODEL
typedef int64_t hsa_signal_value_t;
#else
typedef int32_t hsa_signal_value_t;
#endif

#ifdef __cplusplus
}
#endif
#endif
