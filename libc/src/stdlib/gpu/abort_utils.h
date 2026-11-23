//===-- Internal header for GPU abort -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_GPU_ABORT_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_GPU_ABORT_UTILS_H

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace abort_utils {
[[noreturn]] LIBC_INLINE void abort() {
  // We want to first make sure the server is listening before we abort.
  rpc::Client::Port port = rpc::client.open<LIBC_ABORT>();
  port.send_and_recv([](rpc::Buffer *, uint32_t) {},
                     [](rpc::Buffer *, uint32_t) {});
  port.send([&](rpc::Buffer *, uint32_t) {});

  gpu::end_program();
}
} // namespace abort_utils

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_GPU_ABORT_UTILS_H
