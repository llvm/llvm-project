//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the ungetwc function, which pushes
/// a wide character back into an input stream.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/ungetwc.h"
#include "hdr/types/FILE.h"
#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, ungetwc, (wint_t wc, ::FILE *stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  File *f = reinterpret_cast<File *>(stream);
  auto result = f->ungetwc(wc);
  if (!result.has_value()) {
    libc_errno = result.error();
    return WEOF;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
