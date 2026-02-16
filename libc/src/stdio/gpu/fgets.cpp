//===-- GPU implementation of fgets ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fgets.h"

#include "file.h"
#include "hdr/stdint_proxy.h"
#include "hdr/stdio_macros.h" // for EOF.
#include "hdr/types/FILE.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, fgets,
                   (char *__restrict str, int count,
                    ::FILE *__restrict stream)) {
  if (count < 1)
    return nullptr;

  char *ret = rpc::dispatch<LIBC_READ_FGETS>(
      rpc::client, fgets, rpc::array_ref<char>{str, uint64_t(count)}, count,
      reinterpret_cast<FILE *>(file::from_stream(stream)));

  return ret ? str : nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
