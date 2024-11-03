//===---------- GPU implementation of the external RPC functionion --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/gpu/rpc_reset.h"

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"

namespace __llvm_libc {

// This is the external interface to initialize the RPC client with the
// shared buffer.
LLVM_LIBC_FUNCTION(void, rpc_reset,
                   (unsigned int num_ports, void *rpc_shared_buffer)) {
  __llvm_libc::rpc::client.reset(num_ports, rpc_shared_buffer);
}

} // namespace __llvm_libc
