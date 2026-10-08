//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of fflush.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush.h"

#include "hdr/stdio_macros.h"
#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/fflush_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fflush, (::FILE * stream)) {
  ErrorOr<int> result = stream == nullptr ? internal::flush_all_streams()
                                          : internal::flush_stream(stream);
  if (!result.has_value()) {
    libc_errno = result.error();
    return EOF;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
