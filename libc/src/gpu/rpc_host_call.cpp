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

namespace LIBC_NAMESPACE {

// This calls the associated function pointer on the RPC server with the given
// arguments. We expect that the pointer here is a valid pointer on the server.
LLVM_LIBC_FUNCTION(void, rpc_host_call, (void *fn, void *data, size_t size)) {
  rpc::Client::Port port = rpc::client.open<RPC_HOST_CALL>();
  port.send_n(data, size);
  port.send([=](rpc::Buffer *buffer) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(fn);
  });
  port.recv([](rpc::Buffer *) {});
  port.close();
}

} // namespace LIBC_NAMESPACE
