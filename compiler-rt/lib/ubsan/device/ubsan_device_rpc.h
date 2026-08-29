//===-- ubsan_device_rpc.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_DEVICE_RPC_H
#define UBSAN_DEVICE_RPC_H

#include "hsa.h"

namespace __ubsan {

void StartRpc(hsa_executable_t Exec);
void FlushRpc();
void StopRpc();

} // namespace __ubsan

#endif // UBSAN_DEVICE_RPC_H
