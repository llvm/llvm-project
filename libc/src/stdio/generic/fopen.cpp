//===-- Implementation of fopen -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fopen.h"
#include "src/__support/File/file.h"

#include "hdr/types/FILE.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(::FILE *, fopen,
                   (const char *__restrict name, const char *__restrict mode)) {
  auto result = LIBC_NAMESPACE::openfile(name, mode);
  if (!result.has_value()) {
    libc_errno = result.error();
    return nullptr;
  }
  return reinterpret_cast<::FILE *>(result.value());
}

} // namespace LIBC_NAMESPACE_DECL
