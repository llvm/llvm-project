//===-- GPU Implementation of malloc --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/malloc.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  void *ptr = nullptr;
  rpc::Client::Port port = rpc::client.open<RPC_MALLOC>();
  port.send_and_recv([=](rpc::Buffer *buffer) { buffer->data[0] = size; },
                     [&](rpc::Buffer *buffer) {
                       ptr = reinterpret_cast<void *>(buffer->data[0]);
                     });
  port.close();
  return ptr;
}

} // namespace LIBC_NAMESPACE
