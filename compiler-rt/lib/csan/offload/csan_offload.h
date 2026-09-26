//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal declarations for host-side ConcurrencySanitizer offload reporting.
///
//===----------------------------------------------------------------------===//

#ifndef CSAN_OFFLOAD_H
#define CSAN_OFFLOAD_H

#include "csan_defs.h"
#include "csan_offload_packet.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __csan {

u32 HandleOffloadReport(void *Port, u32 Lanes);

} // namespace __csan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __csan_offload_init();
}

#endif // CSAN_OFFLOAD_H
