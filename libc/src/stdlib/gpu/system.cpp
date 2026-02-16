//===-- GPU implementation of system --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/system.h"

#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, system, (const char *command)) {
  return rpc::dispatch<LIBC_SYSTEM>(rpc::client, system, command);
}

} // namespace LIBC_NAMESPACE_DECL
