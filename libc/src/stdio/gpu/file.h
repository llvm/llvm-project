//===--- GPU helper functions--------------------===//
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

LIBC_INLINE uint64_t write_to_stdout(const void *data, size_t size) {
  uint64_t ret = 0;
  rpc::Client::Port port = rpc::client.open<RPC_WRITE_TO_STDOUT>();
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<uint64_t *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

LIBC_INLINE uint64_t write_to_stderr(const void *data, size_t size) {
  uint64_t ret = 0;
  rpc::Client::Port port = rpc::client.open<RPC_WRITE_TO_STDERR>();
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<uint64_t *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

LIBC_INLINE uint64_t write_to_stream(uintptr_t file, const void *data,
                                     size_t size) {
  uint64_t ret = 0;
  rpc::Client::Port port = rpc::client.open<RPC_WRITE_TO_STREAM>();
  port.send([&](rpc::Buffer *buffer) {
    reinterpret_cast<uintptr_t *>(buffer->data)[0] = file;
  });
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<uint64_t *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

LIBC_INLINE uint64_t write(FILE *f, const void *data, size_t size) {
  if (f == stdout)
    return write_to_stdout(data, size);
  else if (f == stderr)
    return write_to_stderr(data, size);
  else
    return write_to_stream(reinterpret_cast<uintptr_t>(f), data, size);
}

LIBC_INLINE uint64_t read_from_stdin(void *buf, size_t size) {
  uint64_t ret = 0;
  uint64_t recv_size;
  rpc::Client::Port port = rpc::client.open<RPC_READ_FROM_STDIN>();
  port.send([=](rpc::Buffer *buffer) { buffer->data[0] = size; });
  port.recv_n(&buf, &recv_size, [&](uint64_t) { return buf; });
  port.recv([&](rpc::Buffer *buffer) { ret = buffer->data[0]; });
  port.close();
  return ret;
}

LIBC_INLINE uint64_t read_from_stream(uintptr_t file, void *buf, size_t size) {
  uint64_t ret = 0;
  uint64_t recv_size;
  rpc::Client::Port port = rpc::client.open<RPC_READ_FROM_STREAM>();
  port.send([=](rpc::Buffer *buffer) {
    buffer->data[0] = size;
    buffer->data[1] = file;
  });
  port.recv_n(&buf, &recv_size, [&](uint64_t) { return buf; });
  port.recv([&](rpc::Buffer *buffer) { ret = buffer->data[0]; });
  port.close();
  return ret;
}

LIBC_INLINE uint64_t read(FILE *f, void *data, size_t size) {
  if (f == stdin)
    return read_from_stdin(data, size);
  else
    return read_from_stream(reinterpret_cast<uintptr_t>(f), data, size);
}

} // namespace file
} // namespace __llvm_libc
