//===-- Implementation of vsscanf -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vsscanf.h"

#include "src/__support/arg_list.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, vsscanf,
                   (const char *, const char *,
                    va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.

  
  return -1;
}

} // namespace LIBC_NAMESPACE
