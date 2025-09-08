//===-- Implementation of fopencookie -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fopencookie.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/FILE.h"
#include "hdr/types/cookie_io_functions_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/File/cookie_file.h"
#include "src/__support/File/file.h"

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(::FILE *, fopencookie,
                   (void *cookie, const char *mode,
                    cookie_io_functions_t ops)) {
  uint8_t *buffer;
  {
    AllocChecker ac;
    buffer = new (ac) uint8_t[File::DEFAULT_BUFFER_SIZE];
    if (!ac)
      return nullptr;
  }
  AllocChecker ac;
  auto *file =
      new (ac) CookieFile(cookie, ops, buffer, File::DEFAULT_BUFFER_SIZE,
                          _IOFBF, File::mode_flags(mode));
  if (!ac)
    return nullptr;
  return reinterpret_cast<::FILE *>(file);
}

} // namespace LIBC_NAMESPACE_DECL
