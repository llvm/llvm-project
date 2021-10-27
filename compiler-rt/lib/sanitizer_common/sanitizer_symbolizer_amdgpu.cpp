//===-- sanitizer_symbolizer_amdgpu.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#if SANITIZER_AMDGPU
#  include "sanitizer_symbolizer_amdgpu.h"

#  include <dlfcn.h>  //For dlsym

namespace __sanitizer {

static COMgrFunctions comgr = {false};

void getSourceLocation(const char *Result, void *ScopedString) {
  InternalScopedString *ScopedStringObj = (InternalScopedString *)ScopedString;
  ScopedStringObj->append(Result);
}

void AMDGPUCodeObjectSymbolizer::InitCOMgr() {
  if (!comgr.inited_) {
    comgr.create_data =
        (decltype(comgr.create_data))dlsym(RTLD_NEXT, "amd_comgr_create_data");
    comgr.set_data = (decltype(comgr.set_data))dlsym(
        RTLD_NEXT, "amd_comgr_set_data_from_file_slice");
    comgr.create_symbolizer = (decltype(comgr.create_symbolizer))dlsym(
        RTLD_NEXT, "amd_comgr_create_symbolizer_info");
    comgr.symbolize =
        (decltype(comgr.symbolize))dlsym(RTLD_NEXT, "amd_comgr_symbolize");
    comgr.destroy_symbolizer = (decltype(comgr.destroy_symbolizer))dlsym(
        RTLD_NEXT, "amd_comgr_destroy_symbolizer_info");
    comgr.release_data = (decltype(comgr.release_data))dlsym(
        RTLD_NEXT, "amd_comgr_release_data");

    if (!comgr.create_data || !comgr.set_data || !comgr.create_symbolizer ||
        !comgr.symbolize || !comgr.destroy_symbolizer || !comgr.release_data)
      comgr.inited_ = false;
    comgr.inited_ = true;
  }
}

void AMDGPUCodeObjectSymbolizer::Init(int fd, uint64_t off, uint64_t size) {
  InitCOMgr();
  if (comgr.inited_) {
    if (comgr.create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &codeobject))
      return;

    object_cnt = comgr_objects::data;
    if (comgr.set_data(codeobject, fd, off, size)) {
      Release();
      return;
    }

    if (comgr.create_symbolizer(codeobject, &getSourceLocation, &symbolizer)) {
      Release();
      return;
    }

    object_cnt = comgr_objects::data_and_symb;
    init = true;
  }
}

bool AMDGPUCodeObjectSymbolizer::SymbolizePC(uptr addr,
                                             InternalScopedString &source_loc) {
  if (!init)
    return false;
  comgr.symbolize(symbolizer, addr, true, (void *)&source_loc);
  return true;
}

void AMDGPUCodeObjectSymbolizer::Release() {
  // fall-through is avoided to silence warnings.
  switch (object_cnt) {
    case comgr_objects::data_and_symb: {
      comgr.destroy_symbolizer(symbolizer);
      comgr.release_data(codeobject);
      break;
    }
    case comgr_objects::data: {
      comgr.release_data(codeobject);
      break;
    }
    default: {
    }
  }
}
}  // namespace __sanitizer
#endif
