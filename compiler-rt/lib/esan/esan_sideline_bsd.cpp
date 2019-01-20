//===-- esan_sideline_bsd.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// Support for a separate or "sideline" tool thread on FreeBSD.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_FREEBSD

#include "esan_sideline.h"

namespace __esan {

static SidelineThread *TheThread;

bool SidelineThread::launchThread(SidelineFunc takeSample, void *Arg,
                                  u32 FreqMilliSec) {
  return true;
}

bool SidelineThread::joinThread() {
  return true;
}

} // namespace __esan

#endif // SANITIZER_FREEBSD
