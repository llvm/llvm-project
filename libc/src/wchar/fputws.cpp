//===-- Implementation of fputws ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/fputws.h"
#include "hdr/types/FILE.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/string/string_length.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fputws,
                   (const wchar_t *__restrict str, ::FILE *__restrict stream)) {
  LIBC_CRASH_ON_NULLPTR(str);
  LIBC_CRASH_ON_NULLPTR(stream);

  size_t len = internal::string_length(str);

  auto *f = reinterpret_cast<File *>(stream);
  FileIOResult result = f->write(str, len);
  if (result.has_error() || result.value < len) {
    if (result.has_error())
      libc_errno = result.error;
    return -1;
  }

  return static_cast<int>(result.value);
}

} // namespace LIBC_NAMESPACE_DECL
