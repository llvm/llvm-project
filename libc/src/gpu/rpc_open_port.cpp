//===---------- GPU implementation of the external RPC port interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/gpu/rpc_open_port.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

static_assert(sizeof(rpc_port_t) == sizeof(rpc::Client::Port), "ABI mismatch");

LLVM_LIBC_FUNCTION(rpc_port_t, rpc_open_port, (unsigned opcode)) {
  uint32_t uniform = gpu::broadcast_value(gpu::get_lane_mask(), opcode);
  rpc::Client::Port port = rpc::client.open(static_cast<uint16_t>(uniform));
  return cpp::bit_cast<rpc_port_t>(port);
}

} // namespace LIBC_NAMESPACE
