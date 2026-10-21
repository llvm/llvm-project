//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for regcomp.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_REGEX_REGCOMP_H
#define LLVM_LIBC_SRC_REGEX_REGCOMP_H

#include "hdr/types/regex_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int regcomp(regex_t *__restrict preg, const char *__restrict pattern,
            int cflags);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_REGEX_REGCOMP_H
