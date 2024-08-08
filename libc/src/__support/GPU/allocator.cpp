//===-- GPU memory allocator implementation ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allocator.h"

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

void *rpc_allocate(uint64_t size) {
  void *ptr = nullptr;
  rpc::Client::Port port = rpc::client.open<RPC_MALLOC>();
  port.send_and_recv([=](rpc::Buffer *buffer) { buffer->data[0] = size; },
                     [&](rpc::Buffer *buffer) {
                       ptr = reinterpret_cast<void *>(buffer->data[0]);
                     });
  port.close();
  return ptr;
}

void rpc_free(void *ptr) {
  rpc::Client::Port port = rpc::client.open<RPC_FREE>();
  port.send([=](rpc::Buffer *buffer) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(ptr);
  });
  port.close();
}

} // namespace

namespace gpu {

void *allocate(uint64_t size) { return rpc_allocate(size); }

void deallocate(void *ptr) { rpc_free(ptr); }

} // namespace gpu
} // namespace LIBC_NAMESPACE_DECL
