//===-- GPU Implementation of free ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/free.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, free, (void *ptr)) {
  rpc::Client::Port port = rpc::client.open<RPC_FREE>();
  port.send([=](rpc::Buffer *buffer) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(ptr);
  });
  port.close();
}

} // namespace LIBC_NAMESPACE
