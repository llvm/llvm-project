//===-- Shared memory RPC client instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rpc_client.h"
#include "rpc.h"

namespace LIBC_NAMESPACE {
namespace rpc {

/// The libc client instance used to communicate with the server.
Client client;

/// Externally visible symbol to signify the usage of an RPC client to
/// whomever needs to run the server as well as provide a way to initialize
/// the client with a copy..
extern "C" [[gnu::visibility("protected")]] void *LIBC_NAMESPACE_rpc_client =
    &client;

} // namespace rpc
} // namespace LIBC_NAMESPACE
