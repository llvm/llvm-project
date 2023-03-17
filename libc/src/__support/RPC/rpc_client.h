//===-- Shared memory RPC client instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_RPC_RPC_CLIENT_H
#define LLVM_LIBC_SRC_SUPPORT_RPC_RPC_CLIENT_H

#include "rpc.h"

namespace __llvm_libc {
namespace rpc {

/// The libc client instance used to communicate with the server.
extern Client client;

} // namespace rpc
} // namespace __llvm_libc

#endif
