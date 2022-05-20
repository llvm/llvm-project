//===--- amdgpu/dynamic_hsa/hsa.cpp ------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement subset of hsa api by calling into hsa library via dlopen
// Does the dlopen/dlsym calls as part of the call to hsa_init
//
//===----------------------------------------------------------------------===//
#include "hsa.h"
#include "../../../../openmp/libomptarget/include/Debug.h"
#include "../../../../openmp/libomptarget/include/dlwrap.h"

#include <dlfcn.h>

DLWRAP_INITIALIZE();

DLWRAP_INTERNAL(hsa_init, 0);

DLWRAP(hsa_status_string, 2);
DLWRAP(hsa_shut_down, 0);
DLWRAP(hsa_agent_get_info, 3);
DLWRAP(hsa_iterate_agents, 2);

DLWRAP_FINALIZE();

#ifndef DYNAMIC_HSA_PATH
#define DYNAMIC_HSA_PATH "libhsa-runtime64.so.1"
#endif

static bool checkForHSA() {
  // return true if dlopen succeeded and all functions found

  const char *HsaLib = DYNAMIC_HSA_PATH;
  void *DynlibHandle = dlopen(HsaLib, RTLD_NOW);
  if (!DynlibHandle) {
    DP("Unable to load library '%s': %s!\n", HsaLib, dlerror());
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    void *P = dlsym(DynlibHandle, Sym);
    if (P == nullptr) {
      DP("Unable to find '%s' in '%s'!\n", Sym, HsaLib);
      return false;
    }
    DP("Implementing %s with dlsym(%s) -> %p\n", Sym, Sym, P);

    *dlwrap::pointer(I) = P;
  }

  return true;
}

hsa_status_t hsa_init() {
  if (!checkForHSA()) {
    return HSA_STATUS_ERROR;
  }
  return dlwrap_hsa_init();
}
