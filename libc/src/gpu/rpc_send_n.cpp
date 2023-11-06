//===---------- GPU implementation of the external RPC port interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/gpu/rpc_send_n.h"

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

static_assert(sizeof(rpc_port_t) == sizeof(rpc::Client::Port), "ABI mismatch");

LLVM_LIBC_FUNCTION(void, rpc_send_n,
                   (rpc_port_t * handle, const void *src, size_t size)) {
  rpc::Client::Port port = cpp::bit_cast<rpc::Client::Port>(*handle);
  port.send_n(src, size);
  *handle = cpp::bit_cast<rpc_port_t>(port);
}

} // namespace LIBC_NAMESPACE
