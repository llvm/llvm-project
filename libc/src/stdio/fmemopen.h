//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of fmemopen a POSIX function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FMEMOPEN_H
#define LLVM_LIBC_SRC_STDIO_FMEMOPEN_H

#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

::FILE *fmemopen(void *__restrict buf, size_t max_size,
                 const char *__restrict mode);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_FMEMOPEN_H
