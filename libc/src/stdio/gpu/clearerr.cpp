//===-- Implementation of clearerr ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr.h"
#include "file.h"
#include "src/__support/macros/config.h"

#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, clearerr, (::FILE * stream)) {
  rpc::Client::Port port = rpc::client.open<LIBC_CLEARERR>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer, uint32_t) {
        buffer->data[0] = file::from_stream(stream);
      },
      [&](rpc::Buffer *, uint32_t) {});
  port.close();
}

} // namespace LIBC_NAMESPACE_DECL
