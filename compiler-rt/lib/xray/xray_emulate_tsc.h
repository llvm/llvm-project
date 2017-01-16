//===-- xray_emulate_tsc.h --------------------------------------*- C++ -*-===//
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
#ifndef XRAY_EMULATE_TSC_H
#define XRAY_EMULATE_TSC_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "xray_defs.h"
#include <cerrno>
#include <cstdint>
#include <time.h>

namespace __xray {

static constexpr uint64_t NanosecondsPerSecond = 1000ULL * 1000 * 1000;

ALWAYS_INLINE uint64_t readTSC(uint8_t &CPU) XRAY_NEVER_INSTRUMENT {
  timespec TS;
  int result = clock_gettime(CLOCK_REALTIME, &TS);
  if (result != 0) {
    Report("clock_gettime(2) returned %d, errno=%d.", result, int(errno));
    TS.tv_sec = 0;
    TS.tv_nsec = 0;
  }
  CPU = 0;
  return TS.tv_sec * NanosecondsPerSecond + TS.tv_nsec;
}
}

#endif // XRAY_EMULATE_TSC_H
