//===-- Generic RPC server interface --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_LOADER_RPC_H
#define LLVM_LIBC_UTILS_GPU_LOADER_RPC_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stddef.h>

#include "src/__support/RPC/rpc.h"

static __llvm_libc::rpc::Server server;

/// Queries the RPC client at least once and performs server-side work if there
/// are any active requests.
void handle_server() {
  using namespace __llvm_libc;

  // Continue servicing the client until there is no work left and we return.
  for (;;) {
    auto port = server.try_open();
    if (!port)
      return;

    switch (port->get_opcode()) {
    case rpc::Opcode::PRINT_TO_STDERR: {
      uint64_t sizes[rpc::MAX_LANE_SIZE] = {0};
      void *strs[rpc::MAX_LANE_SIZE] = {nullptr};
      port->recv_n(strs, sizes, [&](uint64_t size) { return new char[size]; });
      for (uint64_t i = 0; i < rpc::MAX_LANE_SIZE; ++i) {
        if (strs[i]) {
          fwrite(strs[i], sizes[i], 1, stderr);
          delete[] reinterpret_cast<uint8_t *>(strs[i]);
        }
      }
      break;
    }
    case rpc::Opcode::EXIT: {
      port->recv([](rpc::Buffer *buffer) {
        exit(reinterpret_cast<uint32_t *>(buffer->data)[0]);
      });
      break;
    }
    case rpc::Opcode::TEST_INCREMENT: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        reinterpret_cast<uint64_t *>(buffer->data)[0] += 1;
      });
      break;
    }
    case rpc::Opcode::TEST_INTERFACE: {
      uint64_t cnt = 0;
      bool end_with_recv;
      port->recv([&](rpc::Buffer *buffer) { end_with_recv = buffer->data[0]; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      if (end_with_recv)
        port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      else
        port->send(
            [&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      break;
    }
    case rpc::Opcode::TEST_STREAM: {
      uint64_t sizes[rpc::MAX_LANE_SIZE] = {0};
      void *dst[rpc::MAX_LANE_SIZE] = {nullptr};
      port->recv_n(dst, sizes, [](uint64_t size) { return new char[size]; });
      port->send_n(dst, sizes);
      for (uint64_t i = 0; i < rpc::MAX_LANE_SIZE; ++i) {
        if (dst[i])
          delete[] reinterpret_cast<uint8_t *>(dst[i]);
      }
      break;
    }
    default:
      port->recv([](rpc::Buffer *buffer) {});
    }
    port->close();
  }
}

#endif
