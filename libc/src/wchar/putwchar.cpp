//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for the putwchar function, which
/// writes a single character to stdout.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/putwchar.h"
#include "hdr/types/FILE.h"
#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/stdout.h" // For stdout

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, putwchar, (wchar_t wc)) {
  auto *f = reinterpret_cast<File *>(LIBC_NAMESPACE::stdout);
  FileIOResult result = f->write(&wc, 1);
  if (result.has_error() || result.value < 1) {
    if (result.has_error())
      libc_errno = result.error;
    return WEOF;
  }
  return static_cast<wint_t>(wc);
}

} // namespace LIBC_NAMESPACE_DECL
