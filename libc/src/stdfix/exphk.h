//===-- Implementation header for exphk -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDFIX_EXPHK_H
#define LLVM_LIBC_SRC_STDFIX_EXPHK_H

#include "include/llvm-libc-macros/stdfix-macros.h"

namespace LIBC_NAMESPACE {

short accum exphk(short accum x);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDFIX_EXPHK_H
