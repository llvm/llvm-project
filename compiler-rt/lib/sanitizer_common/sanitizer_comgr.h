//===-- sanitizer_comgr.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// COMgr-backed symbolizer for AMDGPU code objects (SANITIZER_AMDHSA).
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_COMGR_H
#define SANITIZER_COMGR_H

extern "C" {

typedef enum amd_comgr_status_s {
  AMD_COMGR_STATUS_SUCCESS = 0x0,
  AMD_COMGR_STATUS_ERROR = 0x1,
  AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = 0x2,
  AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = 0x3,
} amd_comgr_status_t;

typedef enum amd_comgr_data_kind_s {
  AMD_COMGR_DATA_KIND_UNDEF = 0x0,
  AMD_COMGR_DATA_KIND_SOURCE = 0x1,
  AMD_COMGR_DATA_KIND_INCLUDE = 0x2,
  AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = 0x3,
  AMD_COMGR_DATA_KIND_DIAGNOSTIC = 0x4,
  AMD_COMGR_DATA_KIND_LOG = 0x5,
  AMD_COMGR_DATA_KIND_BC = 0x6,
  AMD_COMGR_DATA_KIND_RELOCATABLE = 0x7,
  AMD_COMGR_DATA_KIND_EXECUTABLE = 0x8,
  AMD_COMGR_DATA_KIND_BYTES = 0x9,
  AMD_COMGR_DATA_KIND_FATBIN = 0x10,
  AMD_COMGR_DATA_KIND_AR = 0x11,
  AMD_COMGR_DATA_KIND_BC_BUNDLE = 0x12,
  AMD_COMGR_DATA_KIND_AR_BUNDLE = 0x13,
  AMD_COMGR_DATA_KIND_OBJ_BUNDLE = 0x14,
  AMD_COMGR_DATA_KIND_SPIRV = 0x15,
  AMD_COMGR_DATA_KIND_LAST = AMD_COMGR_DATA_KIND_SPIRV
} amd_comgr_data_kind_t;

typedef struct amd_comgr_data_s {
  __UINT64_TYPE__ handle;
} amd_comgr_data_t;

typedef struct amd_comgr_symbolizer_info_s {
  __UINT64_TYPE__ handle;
} amd_comgr_symbolizer_info_t;

}  // extern "C"

namespace __sanitizer {

struct COMgrFunctions {
  bool inited_;
  amd_comgr_status_t (*create_data)(amd_comgr_data_kind_t kind,
                                    amd_comgr_data_t* data);
  amd_comgr_status_t (*set_data)(amd_comgr_data_t data, __SIZE_TYPE__ size,
                                 const char* bytes);
  amd_comgr_status_t (*set_data_from_file_slice)(amd_comgr_data_t data, int fd,
                                                 __UINT64_TYPE__ offset,
                                                 __UINT64_TYPE__ size);
  amd_comgr_status_t (*create_symbolizer)(
      amd_comgr_data_t code_object,
      void (*print_symbol_callback)(const char* symbol, void* user_data),
      amd_comgr_symbolizer_info_t* symbolizer_info);
  amd_comgr_status_t (*symbolize)(amd_comgr_symbolizer_info_t symbolizer_info,
                                  __UINT64_TYPE__ address, bool is_code,
                                  void* user_data);
  amd_comgr_status_t (*destroy_symbolizer)(
      amd_comgr_symbolizer_info_t symbolizer_info);
  amd_comgr_status_t (*release_data)(amd_comgr_data_t data);
};

}  // namespace __sanitizer

#endif  // SANITIZER_COMGR_H
