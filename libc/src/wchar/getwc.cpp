//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for the getwc function, which
/// reads a single wide character from the provided stream.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/getwc.h"
#include "hdr/types/FILE.h"
#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, getwc, (::FILE * stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  auto *f = reinterpret_cast<File *>(stream);
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
