//===-- GPU Implementation of fclose --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/__support/macros/config.h"
#include "src/stdio/gpu/file.h"

#include "hdr/stdio_macros.h"
#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fclose, (::FILE * stream)) {
  uint64_t ret = 0;
  uintptr_t file = reinterpret_cast<uintptr_t>(stream);
  rpc::Client::Port port = rpc::client.open<RPC_CLOSE_FILE>();
  port.send_and_recv([=](rpc::Buffer *buffer) { buffer->data[0] = file; },
                     [&](rpc::Buffer *buffer) { ret = buffer->data[0]; });
  port.close();

  if (ret != 0)
    return EOF;
  return static_cast<int>(ret);
}

} // namespace LIBC_NAMESPACE_DECL
