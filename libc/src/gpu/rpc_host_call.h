//===-- Implementation header for RPC functions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_GPU_RPC_HOST_CALL_H
#define LLVM_LIBC_SRC_GPU_RPC_HOST_CALL_H

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE {

void rpc_host_call(void *fn, void *buffer, size_t size);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_GPU_RPC_H_HOST_CALL
