//===-- Implementation header for strstr ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRSTR_H
#define LLVM_LIBC_SRC_STRING_STRSTR_H

namespace __llvm_libc {

char *strstr(const char *haystack, const char *needle);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_STRSTR_H
