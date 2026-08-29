//===-- ubsan_device.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host device reporting runtime for UBSan.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_DEVICE_H
#define UBSAN_DEVICE_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "ubsan_device_packet.h"

namespace __ubsan {

void Initialize();
void PrintDeviceReport(const __ubsan_device_report &R);

// Nest only after RpcMutex, the reverse deadlocks the report thread.
extern __sanitizer::Mutex UbsanDeviceMutex;

} // namespace __ubsan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __ubsan_device_init();
}

#endif // UBSAN_DEVICE_H
