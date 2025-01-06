//===---------- GPU implementation of the external RPC call function ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/gpu/rpc_host_call.h"

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// This calls the associated function pointer on the RPC server with the given
// arguments. We expect that the pointer here is a valid pointer on the server.
LLVM_LIBC_FUNCTION(unsigned long long, rpc_host_call,
                   (void *fn, void *data, size_t size)) {
  rpc::Client::Port port = rpc::client.open<LIBC_HOST_CALL>();
  port.send_n(data, size);
  port.send([=](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(fn);
  });
  unsigned long long ret;
  port.recv([&](rpc::Buffer *buffer, uint32_t) {
    ret = static_cast<unsigned long long>(buffer->data[0]);
  });
  port.close();
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
