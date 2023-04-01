//===-------------- GPU implementation of IO utils --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/string/memory_utils/memcpy_implementations.h"

namespace __llvm_libc {

namespace internal {

static constexpr size_t BUFFER_SIZE = sizeof(rpc::Buffer) - sizeof(uint64_t);
static constexpr size_t MAX_STRING_SIZE = BUFFER_SIZE;

LIBC_INLINE void send_null_terminated(cpp::string_view src) {
  rpc::client.run(
      [&](rpc::Buffer *buffer) {
        buffer->data[0] = rpc::Opcode::PRINT_TO_STDERR;
        char *data = reinterpret_cast<char *>(&buffer->data[1]);
        inline_memcpy(data, src.data(), src.size());
        data[src.size()] = '\0';
      },
      [](rpc::Buffer *) { /* void */ });
}

} // namespace internal

void write_to_stderr(cpp::string_view msg) {
  bool send_empty_string = true;
  for (; !msg.empty();) {
    const auto chunk = msg.substr(0, internal::MAX_STRING_SIZE);
    internal::send_null_terminated(chunk);
    msg.remove_prefix(chunk.size());
    send_empty_string = false;
  }
  if (send_empty_string)
    internal::send_null_terminated("");
}

} // namespace __llvm_libc
