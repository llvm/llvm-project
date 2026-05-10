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

namespace __xray {

bool FindXRaySledSectionInImage(const void *Addr,
                                const XRaySledEntry **SledsBegin,
                                const XRaySledEntry **SledsEnd,
                                const XRayFunctionSledIndex **FnIdxBegin,
                                const XRayFunctionSledIndex **FnIdxEnd)
    XRAY_NEVER_INSTRUMENT {
  Dl_info Info;
  if (!dladdr(Addr, &Info) || !Info.dli_fbase)
    return false;

  const auto *MH =
      reinterpret_cast<const struct mach_header_64 *>(Info.dli_fbase);

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

  if (FnIdx && FnIdxSectionSize > 0) {
    *FnIdxBegin = FnIdx;
    *FnIdxEnd = FnIdx + (FnIdxSectionSize / sizeof(XRayFunctionSledIndex));
  } else {
    *FnIdxBegin = nullptr;
    *FnIdxEnd = nullptr;
  }

  return true;
}

} // namespace __xray

#endif // SANITIZER_APPLE
