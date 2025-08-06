//===-- Implementation header for abort_handler_s ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ANNEX_K_ABORT_HANDLER_S_H
#define LLVM_LIBC_SRC___SUPPORT_ANNEX_K_ABORT_HANDLER_S_H

#include "hdr/stdio_macros.h"
#include "hdr/types/errno_t.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fprintf.h"
#include "src/stdlib/abort.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE static void abort_handler_s(const char *__restrict msg,
                                        void *__restrict ptr, errno_t error) {
  libc_errno = error;
  fprintf(stderr, "abort_handler_s was called in response to a "
                  "runtime-constraint violation.\n\n");
  if (msg)
    fprintf(stderr, "%s\n", msg);
  fprintf(stderr,
          "\n\nNote to end users: This program was terminated as a result\
      of a bug present in the software. Please reach out to your  \
      software's vendor to get more help.\n");

  fflush(stderr);
  abort();
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ANNEX_K_ABORT_HANDLER_S_H
