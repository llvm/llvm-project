//===-- GPU implementation of fgets ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fgets.h"
#include "file.h"
#include "src/__support/macros/config.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"

#include "hdr/stdio_macros.h" // for EOF.
#include "hdr/types/FILE.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, fgets,
                   (char *__restrict str, int count,
                    ::FILE *__restrict stream)) {
  if (count < 1)
    return nullptr;

  uint64_t recv_size;
  void *buf = nullptr;
  rpc::Client::Port port = rpc::client.open<LIBC_READ_FGETS>();
  port.send([=](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = count;
    buffer->data[1] = file::from_stream(stream);
  });
  port.recv_n(&buf, &recv_size,
              [&](uint64_t) { return reinterpret_cast<void *>(str); });
  port.close();

  if (recv_size == 0)
    return nullptr;

  return str;
}

} // namespace LIBC_NAMESPACE_DECL
