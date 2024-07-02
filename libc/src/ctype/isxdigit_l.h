//===-- Implementation header for isxdigit_l ----------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_CTYPE_ISXDIGIT_L_H
#define LLVM_LIBC_SRC_CTYPE_ISXDIGIT_L_H

#include "include/llvm-libc-types/locale_t.h"

namespace LIBC_NAMESPACE {

int isxdigit_l(int c, locale_t);

} // namespace LIBC_NAMESPACE

#endif //  LLVM_LIBC_SRC_CTYPE_ISXDIGIT_L_H
