//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for freopen.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FREOPEN_H
#define LLVM_LIBC_SRC_STDIO_FREOPEN_H

#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

::FILE *freopen(const char *__restrict filename, const char *__restrict mode,
                ::FILE *__restrict stream);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_FREOPEN_H
