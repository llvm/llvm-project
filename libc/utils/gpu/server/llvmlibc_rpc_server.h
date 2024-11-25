//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_SERVER_RPC_SERVER_H
#define LLVM_LIBC_UTILS_GPU_SERVER_RPC_SERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int libc_handle_rpc_port(void *port, uint32_t num_lanes);

#ifdef __cplusplus
}
#endif

#endif
