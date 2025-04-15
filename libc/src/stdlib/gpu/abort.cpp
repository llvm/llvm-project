//===-- GPU implementation of abort ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "src/stdlib/abort.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, abort, ()) {
  // We want to first make sure the server is listening before we abort.
  rpc::Client::Port port = rpc::client.open<LIBC_ABORT>();
  port.send_and_recv([](rpc::Buffer *, uint32_t) {},
                     [](rpc::Buffer *, uint32_t) {});
  port.send([&](rpc::Buffer *, uint32_t) {});
  port.close();

  gpu::end_program();
}

} // namespace LIBC_NAMESPACE_DECL
