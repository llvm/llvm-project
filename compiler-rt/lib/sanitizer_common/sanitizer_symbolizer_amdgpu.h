//===-- sanitizer_symbolizer_fuchsia.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef SANITIZER_SYMBOLIZER_AMDGPU_H
#define SANITIZER_SYMBOLIZER_AMDGPU_H

#if SANITIZER_AMDGPU
#  include "sanitizer_common.h"
#  include "sanitizer_symbolizer_internal.h"
#  ifdef AMD_COMGR
#    include "amd_comgr.h"
#  else
// Note: We require amd_comgr.h.in in COMgr source directory to be valid C
// header file since we have a circular dependency between compiler-rt and COMgr
// builds.
// FIXME: Find a long-term solution for resolving this circular dependency and
// avoid including amd_comgr.h.in.
#    include "amd_comgr.h.in"
#  endif

namespace __sanitizer {

struct COMgrFunctions {
  bool inited_;
  amd_comgr_status_t (*create_data)(amd_comgr_data_kind_t data_type,
                                    amd_comgr_data_t *data_handle);
  amd_comgr_status_t (*set_data)(amd_comgr_data_t data_handle, int fd,
                                 uint64_t offset, uint64_t size);
  amd_comgr_status_t (*create_symbolizer)(
      amd_comgr_data_t object_handle, void (*callback)(const char *, void *),
      amd_comgr_symbolizer_info_t *symbolizer_object);
  amd_comgr_status_t (*symbolize)(amd_comgr_symbolizer_info_t symbolizer_handle,
                                  uint64_t addr, bool iscode, void *data);
  amd_comgr_status_t (*destroy_symbolizer)(
      amd_comgr_symbolizer_info_t symbolizer_handle);
  amd_comgr_status_t (*release_data)(amd_comgr_data_t data_handle);
};

// Symbolizer for AMDGPU CodeObject.
class AMDGPUCodeObjectSymbolizer {
 public:
  AMDGPUCodeObjectSymbolizer() : object_cnt(comgr_objects::no_objs) {}

  void Init(int fd, uint64_t offset, uint64_t size);
  bool SymbolizePC(uptr addr, InternalScopedString &source_loc);
  void Release();

 private:
  void InitCOMgr();
  amd_comgr_data_t codeobject;
  amd_comgr_symbolizer_info_t symbolizer;
  enum comgr_objects { no_objs = 0, data = 1, data_and_symb = 2 } object_cnt;
  bool init = false;
};
}  // namespace __sanitizer
#endif
#endif  // SANITIZER_SYMBOLIZER_AMDGPU_H
