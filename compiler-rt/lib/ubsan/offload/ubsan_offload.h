//===-- ubsan_offload.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host offload reporting runtime for UBSan.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_OFFLOAD_H
#define UBSAN_OFFLOAD_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "ubsan_offload_packet.h"

namespace __ubsan {

void Initialize();
void PrintOffloadReport(const __ubsan_offload_report &R);

// Nest only after RpcMutex, the reverse deadlocks the report thread.
extern __sanitizer::Mutex UbsanOffloadMutex;

} // namespace __ubsan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __ubsan_offload_init();
}

#endif // UBSAN_OFFLOAD_H
