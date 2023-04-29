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

static __llvm_libc::cpp::Atomic<uint32_t> lock;

/// Queries the RPC client at least once and performs server-side work if there
/// are any active requests.
void handle_server() {
  auto port = server.try_open();
  if (!port)
    return;

  switch (port->get_opcode()) {
  case __llvm_libc::rpc::Opcode::PRINT_TO_STDERR: {
    uint64_t str_size;
    char *str = nullptr;
    port->recv_n([&](uint64_t size) {
      str_size = size;
      str = new char[size];
      return str;
    });
    fwrite(str, str_size, 1, stderr);
    delete[] str;
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
    port->recv([](__llvm_libc::rpc::Buffer *) { /* no-op */ });
    return;
  }
  port->close();
}
#endif
