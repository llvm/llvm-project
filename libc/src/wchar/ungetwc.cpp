//===-- Implementation of ungetwc -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/ungetwc.h"
#include "hdr/types/FILE.h"
#include "hdr/types/wint_t.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, ungetwc, (wint_t wc, ::FILE *stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);

  auto *f = reinterpret_cast<File *>(stream);
  return f->ungetwc(wc);
}

} // namespace LIBC_NAMESPACE_DECL
