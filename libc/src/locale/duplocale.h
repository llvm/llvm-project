//===-- Implementation header for duplocale ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LOCALE_DUPLOCALE_H
#define LLVM_LIBC_SRC_LOCALE_DUPLOCALE_H

#include "include/llvm-libc-types/locale_t.h"

namespace LIBC_NAMESPACE {

locale_t duplocale(locale_t);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_LINK_DL_ITERATE_PHDR_H
