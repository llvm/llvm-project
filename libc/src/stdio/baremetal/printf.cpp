//===-- Implementation of printf for baremetal ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf.h"

#include "src/__support/File/baremetal/file.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/vfprintf_internal.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, printf, (const char *__restrict format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  va_end(vlist);

  return vfprintf_internal(stdout, format, args);
}

} // namespace LIBC_NAMESPACE_DECL
