//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the fputwc function, which writes a
/// single wide character to a file.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/fputwc.h"
#include "hdr/types/FILE.h"
#include "hdr/types/wchar_t.h"
#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, fputwc, (wchar_t wc, ::FILE *stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  auto *f = reinterpret_cast<File *>(stream);
  FileIOResult result = f->write(&wc, 1);
  if (result.has_error()) {
    libc_errno = result.error;
    return WEOF;
  }
  // It should be impossible for result.value (the number of characters written)
  // to be any value other than 1 at this point. If the system failed to write
  // then f->write should have returned an error.
  LIBC_ASSERT(result.value == 1);
  return static_cast<wint_t>(wc);
}

} // namespace LIBC_NAMESPACE_DECL
