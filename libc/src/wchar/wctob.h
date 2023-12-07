//===-- Implementation header for wctob -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_WCTOB_H
#define LLVM_LIBC_SRC_WCHAR_WCTOB_H

#include <wchar.h>

namespace __llvm_libc {

int wctob(wint_t c);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_WCHAR_WCTOB_H
