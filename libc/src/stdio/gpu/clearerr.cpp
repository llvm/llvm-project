//===-- Implementation of clearerr ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr.h"
#include "file.h"

#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, clearerr, (::FILE * stream)) {
  rpc::Client::Port port = rpc::client.open<RPC_CLEARERR>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer) { buffer->data[0] = file::from_stream(stream); },
      [&](rpc::Buffer *) {});
  port.close();
}

} // namespace __llvm_libc
