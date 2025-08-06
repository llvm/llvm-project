//===-- Implementation header of fopen_s ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FOPEN_S_H
#define LLVM_LIBC_SRC_STDIO_FOPEN_S_H

#include "hdr/types/FILE.h"
#include "include/llvm-libc-types/errno_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

errno_t fopen_s(FILE *__restrict *__restrict streamptr,
                const char *__restrict filename, const char *__restrict mode);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_FOPEN_S_H
