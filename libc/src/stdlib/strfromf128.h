//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Function declaration of strfromf128.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_STRFROMF128_H
#define LLVM_LIBC_SRC_STDLIB_STRFROMF128_H

#include "hdr/types/size_t.h"
#include "include/llvm-libc-types/float128.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int strfromf128(char *__restrict s, size_t n, const char *__restrict format,
                float128 fp);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_STRFROMF128_H
