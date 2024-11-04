//===-- GPU Implementation of rename --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/rename.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/stdio/gpu/file.h"

#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, rename, (const char *oldpath, const char *newpath)) {
  int ret;
  rpc::Client::Port port = rpc::client.open<RPC_RENAME>();
  port.send_n(oldpath, internal::string_length(oldpath) + 1);
  port.send_n(newpath, internal::string_length(newpath) + 1);
  port.recv(
      [&](rpc::Buffer *buffer) { ret = static_cast<int>(buffer->data[0]); });
  port.close();

  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
