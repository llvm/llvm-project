//===-- GPU Implementation of fopen ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fopen.h"

#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/stdio/gpu/file.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(::FILE *, fopen,
                   (const char *__restrict path, const char *__restrict mode)) {
  uintptr_t file;
  rpc::Client::Port port = rpc::client.open<LIBC_OPEN_FILE>();
  port.send_n(path, internal::string_length(path) + 1);
  port.send_and_recv(
      [=](rpc::Buffer *buffer, uint32_t) {
        inline_memcpy(buffer->data, mode, internal::string_length(mode) + 1);
      },
      [&](rpc::Buffer *buffer, uint32_t) { file = buffer->data[0]; });
  port.close();

  return reinterpret_cast<FILE *>(file);
}

} // namespace LIBC_NAMESPACE_DECL
