//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for the getwchar function, which
/// reads a single character from stdin.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/getwchar.h"
#include "hdr/types/FILE.h"
#include "hdr/types/wchar_t.h"
#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/stdin.h" // For stdin

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, getwchar, ()) {
  auto *f = reinterpret_cast<File *>(LIBC_NAMESPACE::stdin);
  wchar_t wc;
  FileIOResult result = f->read(&wc, 1);
  if (result.has_error() || result.value < 1) {
    if (result.has_error())
      libc_errno = result.error;
    return WEOF;
  }
  return static_cast<wint_t>(wc);
}

} // namespace LIBC_NAMESPACE_DECL
