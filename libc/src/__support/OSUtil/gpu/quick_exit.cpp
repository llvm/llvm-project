//===---------- GPU implementation of a quick exit function -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_GPU_QUICK_EXIT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_GPU_QUICK_EXIT_H

#include "quick_exit.h"

#include "src/__support/RPC/rpc_client.h"
#include "src/__support/macros/properties/architectures.h"

namespace __llvm_libc {

void quick_exit(int status) {
  // We want to first make sure the server is listening before we exit.
  rpc::Client::Port port = rpc::client.open<RPC_EXIT>();
  port.send_and_recv([](rpc::Buffer *) {}, [](rpc::Buffer *) {});
  port.send([&](rpc::Buffer *buffer) {
    reinterpret_cast<uint32_t *>(buffer->data)[0] = status;
  });
  port.close();

  gpu::end_program();
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_GPU_QUICK_EXIT_H
