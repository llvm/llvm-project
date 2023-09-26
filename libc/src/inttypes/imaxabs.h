//===-- Implementation header for imaxabs -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_INTTYPES_IMAXABS_H
#define LLVM_LIBC_SRC_INTTYPES_IMAXABS_H

#include <inttypes.h>

namespace LIBC_NAMESPACE {

intmax_t imaxabs(intmax_t j);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_INTTYPES_IMAXABS_H
