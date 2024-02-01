//===-- Implementation header for imaxdiv -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_INTTYPES_IMAXDIV_H
#define LLVM_LIBC_SRC_INTTYPES_IMAXDIV_H

#include <inttypes.h>

namespace LIBC_NAMESPACE {

imaxdiv_t imaxdiv(intmax_t x, intmax_t y);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_INTTYPES_IMAXDIV_H
