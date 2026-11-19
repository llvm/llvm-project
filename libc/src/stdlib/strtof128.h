//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Function declaration of strtof128.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_STRTOF128_H
#define LLVM_LIBC_SRC_STDLIB_STRTOF128_H

#include "include/llvm-libc-types/float128.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: Remove this alias once transition to emulated float128 is complete.
#if !defined(LIBC_TYPES_HAS_NATIVE_FLOAT128)
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // !LIBC_TYPES_HAS_NATIVE_FLOAT128

float128 strtof128(const char *__restrict str, char **__restrict str_end);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_STRTOF128_H
