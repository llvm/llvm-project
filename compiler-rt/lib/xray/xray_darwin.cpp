//===-- xray_darwin.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Darwin-specific XRay section discovery using getsectiondata().
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_APPLE

#include "xray_defs.h"
#include "xray_interface_internal.h"

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <mach-o/loader.h>

namespace __xray {

static bool GetXRaySledSections(
    const struct mach_header_64 *MH, const XRaySledEntry **SledsBegin,
    const XRaySledEntry **SledsEnd, const XRayFunctionSledIndex **FnIdxBegin,
    const XRayFunctionSledIndex **FnIdxEnd) XRAY_NEVER_INSTRUMENT {
  unsigned long SledSectionSize = 0;
  const auto *Sleds = reinterpret_cast<const XRaySledEntry *>(
      getsectiondata(MH, "__DATA", "xray_instr_map", &SledSectionSize));

  if (!Sleds || SledSectionSize == 0)
    return false;

  *SledsBegin = Sleds;
  *SledsEnd = Sleds + (SledSectionSize / sizeof(XRaySledEntry));

  unsigned long FnIdxSectionSize = 0;
  const auto *FnIdx = reinterpret_cast<const XRayFunctionSledIndex *>(
      getsectiondata(MH, "__DATA", "xray_fn_idx", &FnIdxSectionSize));

  *FnIdxBegin = FnIdx;
  *FnIdxEnd = FnIdx ? FnIdx + (FnIdxSectionSize / sizeof(XRayFunctionSledIndex))
                    : nullptr;

  return true;
}

bool FindXRaySledSectionInImage(
    const void *Addr, const XRaySledEntry **SledsBegin,
    const XRaySledEntry **SledsEnd, const XRayFunctionSledIndex **FnIdxBegin,
    const XRayFunctionSledIndex **FnIdxEnd) XRAY_NEVER_INSTRUMENT {
  Dl_info Info;
  if (!dladdr(Addr, &Info) || !Info.dli_fbase)
    return false;

  const auto *MH =
      reinterpret_cast<const struct mach_header_64 *>(Info.dli_fbase);
  return GetXRaySledSections(MH, SledsBegin, SledsEnd, FnIdxBegin, FnIdxEnd);
}

static void XRayDyldImageAdded(const struct mach_header *MH,
                               intptr_t Slide) XRAY_NEVER_INSTRUMENT {
  const auto *MH64 = reinterpret_cast<const struct mach_header_64 *>(MH);

  const XRaySledEntry *SledsBegin = nullptr;
  const XRaySledEntry *SledsEnd = nullptr;
  const XRayFunctionSledIndex *FnIdxBegin = nullptr;
  const XRayFunctionSledIndex *FnIdxEnd = nullptr;

  if (!GetXRaySledSections(MH64, &SledsBegin, &SledsEnd, &FnIdxBegin,
                           &FnIdxEnd))
    return;

  bool IsDSO = (MH64->filetype != MH_EXECUTE);
  __xray_register_sleds(SledsBegin, SledsEnd, FnIdxBegin, FnIdxEnd, IsDSO, {});
}

void RegisterDyldImageCallback() XRAY_NEVER_INSTRUMENT {
  _dyld_register_func_for_add_image(XRayDyldImageAdded);
}

} // namespace __xray

#endif // SANITIZER_APPLE
