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

static __llvm_libc::cpp::Atomic<uint32_t>
    lock[__llvm_libc::rpc::default_port_count] = {0};

/// Queries the RPC client at least once and performs server-side work if there
/// are any active requests.
void handle_server() {
  // Continue servicing the client until there is no work left and we return.
  for (;;) {
    auto port = server.try_open();
    if (!port)
      return;

    switch (port->get_opcode()) {
    case __llvm_libc::rpc::Opcode::PRINT_TO_STDERR: {
      uint64_t str_size[__llvm_libc::rpc::MAX_LANE_SIZE] = {0};
      char *strs[__llvm_libc::rpc::MAX_LANE_SIZE] = {nullptr};
      port->recv_n([&](uint64_t size, uint32_t id) {
        str_size[id] = size;
        strs[id] = new char[size];
        return strs[id];
      });
      for (uint64_t i = 0; i < __llvm_libc::rpc::MAX_LANE_SIZE; ++i) {
        if (strs[i]) {
          fwrite(strs[i], str_size[i], 1, stderr);
          delete[] strs[i];
        }
      }
      break;
    }
    case __llvm_libc::rpc::Opcode::EXIT: {
      port->recv([](__llvm_libc::rpc::Buffer *buffer) {
        exit(reinterpret_cast<uint32_t *>(buffer->data)[0]);
      });
      break;
    }
    case __llvm_libc::rpc::Opcode::TEST_INCREMENT: {
      port->recv_and_send([](__llvm_libc::rpc::Buffer *buffer) {
        reinterpret_cast<uint64_t *>(buffer->data)[0] += 1;
      });
      break;
    }
    default:
      port->recv([](__llvm_libc::rpc::Buffer *buffer) {});
    }
    port->close();
  }
}
#endif
