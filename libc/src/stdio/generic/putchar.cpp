//===-- Implementation of putchar -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/putchar.h"
#include "src/__support/File/file.h"

#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, putchar, (int c)) {
  unsigned char uc = static_cast<unsigned char>(c);

  auto result = LIBC_NAMESPACE::stdout->write(&uc, 1);
  if (result.has_error())
    libc_errno = result.error;
  size_t written = result.value;

  if (1 != written) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
