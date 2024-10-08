//===-- GPU implementation of system --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

#include "src/stdlib/system.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, system, (const char *command)) {
  int ret;
  rpc::Client::Port port = rpc::client.open<RPC_SYSTEM>();
  port.send_n(command, internal::string_length(command) + 1);
  port.recv(
      [&](rpc::Buffer *buffer) { ret = static_cast<int>(buffer->data[0]); });
  port.close();

  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
