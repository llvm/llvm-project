//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the swprintf function.
///
//===----------------------------------------------------------------------===//

#include "src/wchar/swprintf.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, swprintf,
                   (wchar_t *__restrict buffer, size_t bufsz,
                    const wchar_t *__restrict format, ...)) {
  (void)buffer;
  (void)bufsz;
  (void)format;
  // Always returns -1 for now
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
