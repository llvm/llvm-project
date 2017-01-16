//===-- xray_x86_64.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_X86_64_H
#define XRAY_X86_64_H

#include <cstdint>
#include <x86intrin.h>

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "xray_defs.h"

namespace __xray {

ALWAYS_INLINE uint64_t readTSC(uint8_t &CPU) XRAY_NEVER_INSTRUMENT {
  unsigned LongCPU;
  uint64_t TSC = __rdtscp(&LongCPU);
  CPU = LongCPU;
  return TSC;
}
}

#endif // XRAY_X86_64_H
