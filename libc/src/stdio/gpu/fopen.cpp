//===-- GPU Implementation of fopen ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fopen.h"
#include "src/__support/CPP/string_view.h"
#include "src/stdio/gpu/file.h"

#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(::FILE *, fopen,
                   (const char *__restrict path, const char *__restrict mode)) {
  uintptr_t file;
  rpc::Client::Port port = rpc::client.open<RPC_OPEN_FILE>();
  port.send_n(path, internal::string_length(path) + 1);
  port.send_and_recv(
      [=](rpc::Buffer *buffer) {
        inline_memcpy(buffer->data, mode, internal::string_length(mode) + 1);
      },
      [&](rpc::Buffer *buffer) { file = buffer->data[0]; });
  port.close();

  return reinterpret_cast<FILE *>(file);
}

} // namespace __llvm_libc
