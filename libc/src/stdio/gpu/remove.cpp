//===-- Implementation of remove ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/remove.h"
#include "file.h"
#include "src/__support/macros/config.h"

#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, remove, (const char *path)) {
  int ret;
  rpc::Client::Port port = rpc::client.open<LIBC_REMOVE>();
  port.send_n(path, internal::string_length(path) + 1);
  port.recv([&](rpc::Buffer *buffer, uint32_t) {
    ret = static_cast<int>(buffer->data[0]);
  });
  port.close();
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
