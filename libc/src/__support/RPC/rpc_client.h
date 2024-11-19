//===-- Shared memory RPC client instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_RPC_CLIENT_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_RPC_CLIENT_H

#include "rpc.h"

#include "include/llvm-libc-types/rpc_opcodes_t.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace rpc {

static_assert(cpp::is_trivially_copyable<Client>::value &&
                  sizeof(Process<true>) == sizeof(Process<false>),
              "The client is not trivially copyable from the server");

/// The libc client instance used to communicate with the server.
extern Client client;

} // namespace rpc
} // namespace LIBC_NAMESPACE_DECL

#endif
