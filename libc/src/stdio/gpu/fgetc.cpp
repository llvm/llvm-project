//===-- GPU implementation of fgetc ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fgetc.h"
#include "file.h"
#include "src/__support/macros/config.h"

#include "hdr/stdio_macros.h" // for EOF.
#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fgetc, (::FILE * stream)) {
  unsigned char c;
  size_t r = file::read(stream, &c, 1);

  if (r != 1)
    return EOF;
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
