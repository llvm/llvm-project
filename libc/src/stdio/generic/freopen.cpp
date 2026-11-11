//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of freopen.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/freopen.h"
#include "src/__support/File/file.h"

#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(::FILE *, freopen,
                   (const char *__restrict filename,
                    const char *__restrict mode, ::FILE *__restrict stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  LIBC_CRASH_ON_NULLPTR(mode);

  auto *file = reinterpret_cast<File *>(stream);

  int error = reopenfile(file, filename, mode);

  if (error != 0) {
    libc_errno = error;
    return nullptr;
  }

  return stream;
}

} // namespace LIBC_NAMESPACE_DECL
