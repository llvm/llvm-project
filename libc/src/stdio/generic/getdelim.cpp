//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getdelim.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/getdelim.h"

#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/stdio/generic/getdelim_impl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, getdelim,
                   (char **__restrict lineptr, size_t *__restrict n, int delim,
                    ::FILE *__restrict stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);
  auto result = internal::getdelim_impl(lineptr, n, delim,
                                        reinterpret_cast<File *>(stream));
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
