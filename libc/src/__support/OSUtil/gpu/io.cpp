//===-------------- GPU implementation of IO utils --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io.h"

#include "src/__support/RPC/rpc_client.h"
#include "src/string/string_utils.h"

namespace __llvm_libc {

void write_to_stderr(const char *msg) {
  uint64_t length = internal::string_length(msg) + 1;
  uint64_t buffer_len = sizeof(rpc::Buffer) - sizeof(uint64_t);
  for (uint64_t i = 0; i < length; i += buffer_len) {
    rpc::client.run(
        [&](rpc::Buffer *buffer) {
          buffer->data[0] = rpc::Opcode::PRINT_TO_STDERR;
          inline_memcpy(reinterpret_cast<char *>(&buffer->data[1]), &msg[i],
                        (length > buffer_len ? buffer_len : length));
        },
        [](rpc::Buffer *) { /* void */ });
  }
}

} // namespace __llvm_libc
