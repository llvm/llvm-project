//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Formats a report to be presented to the user.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_REPORT_H
#define DASAN_REPORT_H

#include "dasan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __dasan {

using namespace __sanitizer;

u32 HandleDeviceReport(void* Port, u32 Lanes);
bool HandleHostReport(uptr Addr, uptr AccessSize, uptr Base, uptr AllocSize,
                      bool IsWrite, bool Recover, uptr Pc, uptr Bp);

void ReportInvalidFree(uptr Addr, u32 Stack, u32 Tid);

void PrintStats();

}  // namespace __dasan

#endif  // DASAN_REPORT_H
