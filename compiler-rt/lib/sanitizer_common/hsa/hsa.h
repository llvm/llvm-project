//===-- hsa/hsa.h ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal HSA API declarations used by compiler-rt host sanitizers.
// Matches ROCr layout/ABI; does not require ROCm headers at build time.
//
//===----------------------------------------------------------------------===//

#ifndef HSA_H_
#define HSA_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HSA_STATUS_SUCCESS = 0x0,
  HSA_STATUS_INFO_BREAK = 0x1,
  HSA_STATUS_ERROR = 0x1000,
  HSA_STATUS_ERROR_INVALID_ARGUMENT = 0x1001,
  HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,
  HSA_STATUS_ERROR_OUT_OF_RESOURCES = 0x100C,
  HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 0x1025,
} hsa_status_t;

typedef struct hsa_agent_s {
  uint64_t handle;
} hsa_agent_t;

typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;

hsa_status_t hsa_init(void);
hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size);

#ifdef __cplusplus
}
#endif

#endif  // HSA_H_
