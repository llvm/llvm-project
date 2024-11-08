//===-- Implementation of setbuf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/setbuf.h"
#include "hdr/stdio_macros.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, setbuf,
                   (::FILE *__restrict stream, char *__restrict buf)) {
  int mode = _IOFBF;
  if (buf == nullptr)
    mode = _IONBF;
  int err = reinterpret_cast<LIBC_NAMESPACE::File *>(stream)->set_buffer(
      buf, BUFSIZ, mode);
  if (err != 0)
    libc_errno = err;
}

} // namespace LIBC_NAMESPACE_DECL
