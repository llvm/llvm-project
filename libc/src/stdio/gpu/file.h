//===--- GPU helper functions for file I/O using RPC ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc_client.h"
#include "src/string/string_utils.h"

#include <stdio.h>

namespace __llvm_libc {
namespace file {

template <uint16_t opcode>
LIBC_INLINE uint64_t write_impl(::FILE *file, const void *data, size_t size) {
  uint64_t ret = 0;
  rpc::Client::Port port = rpc::client.open<opcode>();

  if constexpr (opcode == RPC_WRITE_TO_STREAM) {
    port.send([&](rpc::Buffer *buffer) {
      buffer->data[0] = reinterpret_cast<uintptr_t>(file);
    });
  }

  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<uint64_t *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

LIBC_INLINE uint64_t write(::FILE *f, const void *data, size_t size) {
  if (f == stdout)
    return write_impl<RPC_WRITE_TO_STDOUT>(f, data, size);
  else if (f == stderr)
    return write_impl<RPC_WRITE_TO_STDERR>(f, data, size);
  else
    return write_impl<RPC_WRITE_TO_STREAM>(f, data, size);
}

template <uint16_t opcode>
LIBC_INLINE uint64_t read_from_stream(::FILE *file, void *buf, size_t size) {
  uint64_t ret = 0;
  uint64_t recv_size;
  rpc::Client::Port port = rpc::client.open<opcode>();
  port.send([=](rpc::Buffer *buffer) {
    buffer->data[0] = size;
    if constexpr (opcode == RPC_READ_FROM_STREAM)
      buffer->data[1] = reinterpret_cast<uintptr_t>(file);
  });
  port.recv_n(&buf, &recv_size, [&](uint64_t) { return buf; });
  port.recv([&](rpc::Buffer *buffer) { ret = buffer->data[0]; });
  port.close();
  return ret;
}

LIBC_INLINE uint64_t read(::FILE *f, void *data, size_t size) {
  if (f == stdin)
    return read_from_stream<RPC_READ_FROM_STDIN>(f, data, size);
  else
    return read_from_stream<RPC_READ_FROM_STREAM>(f, data, size);
}

} // namespace file
} // namespace __llvm_libc
