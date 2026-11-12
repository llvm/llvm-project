//===-- Implementation of fread for baremetal -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fread.h"

#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/file_internal.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, fread,
                   (void *__restrict buffer, size_t size, size_t nmemb,
                    ::FILE *stream)) {
  if (size == 0 || nmemb == 0)
    return 0;
  auto result =
      read_internal(reinterpret_cast<char *>(buffer), size * nmemb, stream);
  if (result.has_error())
    libc_errno = result.error;
  LIBC_ASSERT(result.value % size == 0 && "result not multiple of size");
  return result.value / size;
}

} // namespace LIBC_NAMESPACE_DECL
