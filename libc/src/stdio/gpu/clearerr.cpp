//===-- Implementation of clearerr ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr.h"
#include "src/__support/RPC/rpc_client.h"

#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, clearerr, (::FILE * stream)) {
  reinterpret_cast<__llvm_libc::File *>(stream)->clearerr();
  rpc::Client::Port port = rpc::client.open<RPC_FERROR>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer) {
        buffer->data[0] = reinterpret_cast<uintptr_t>(stream);
      },
      [&](rpc::Buffer *) {});
  port.close();
  return ret;
}

} // namespace __llvm_libc
