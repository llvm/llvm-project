//===-- Shared memory RPC client instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rpc.h"

namespace __llvm_libc {
namespace rpc {

/// The libc client instance used to communicate with the server.
Client client;

/// Externally visible symbol to signify the usage of an RPC client to
/// whomever needs to run the server.
extern "C" [[gnu::visibility("protected")]] const bool __llvm_libc_rpc = false;

} // namespace rpc
} // namespace __llvm_libc
