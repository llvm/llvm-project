//===-- xray_init.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// XRay initialisation logic for DSOs.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_atomic.h"
#include "xray_defs.h"
#include "xray_flags.h"
#include "xray_interface_internal.h"

using namespace __sanitizer;

extern "C" {
#if !SANITIZER_APPLE
extern const XRaySledEntry __start_xray_instr_map[] __attribute__((weak))
__attribute__((visibility("hidden")));
extern const XRaySledEntry __stop_xray_instr_map[] __attribute__((weak))
__attribute__((visibility("hidden")));
extern const XRayFunctionSledIndex __start_xray_fn_idx[] __attribute__((weak))
__attribute__((visibility("hidden")));
extern const XRayFunctionSledIndex __stop_xray_fn_idx[] __attribute__((weak))
__attribute__((visibility("hidden")));
#endif
}

// Handler functions to call in the patched entry/exit sled.
extern atomic_uintptr_t XRayPatchedFunction;
extern atomic_uintptr_t XRayArgLogger;
extern atomic_uintptr_t XRayPatchedCustomEvent;
extern atomic_uintptr_t XRayPatchedTypedEvent;

static int __xray_object_id{-1};

// Note: .preinit_array initialization does not work for DSOs
__attribute__((constructor(0))) static void
__xray_init_dso() XRAY_NEVER_INSTRUMENT {
#if SANITIZER_APPLE
  const XRaySledEntry *SledsBegin = nullptr;
  const XRaySledEntry *SledsEnd = nullptr;
  const XRayFunctionSledIndex *FnIdxBegin = nullptr;
  const XRayFunctionSledIndex *FnIdxEnd = nullptr;

  if (!__xray::FindXRaySledSectionInImage(
          reinterpret_cast<const void *>(&__xray_init_dso), &SledsBegin,
          &SledsEnd, &FnIdxBegin, &FnIdxEnd))
    return;

  __xray_object_id =
      __xray_register_dso(SledsBegin, SledsEnd, FnIdxBegin, FnIdxEnd, {});
#else
  // Register sleds in main XRay runtime.
  __xray_object_id =
      __xray_register_dso(__start_xray_instr_map, __stop_xray_instr_map,
                          __start_xray_fn_idx, __stop_xray_fn_idx, {});
#endif
}

__attribute__((destructor(0))) static void
__xray_finalize_dso() XRAY_NEVER_INSTRUMENT {
  // Inform the main runtime that this DSO is no longer used.
  __xray_deregister_dso(__xray_object_id);
}
