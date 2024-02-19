//===-- GPU implementation of fseek ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fseek.h"
#include "file.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, fseek, (::FILE * stream, long offset, int whence)) {
  int ret;
  rpc::Client::Port port = rpc::client.open<RPC_FSEEK>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer) {
        buffer->data[0] = file::from_stream(stream);
        buffer->data[1] = static_cast<uint64_t>(offset);
        buffer->data[2] = static_cast<uint64_t>(whence);
      },
      [&](rpc::Buffer *buffer) { ret = static_cast<int>(buffer->data[0]); });
  port.close();
  return ret;
}

} // namespace LIBC_NAMESPACE
