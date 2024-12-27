//===-- Implementation of ferror ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/ferror.h"
#include "file.h"
#include "src/__support/macros/config.h"

#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ferror, (::FILE * stream)) {
  int ret;
  rpc::Client::Port port = rpc::client.open<LIBC_FERROR>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer, uint32_t) {
        buffer->data[0] = file::from_stream(stream);
      },
      [&](rpc::Buffer *buffer, uint32_t) {
        ret = static_cast<int>(buffer->data[0]);
      });
  port.close();
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
