//===-- Implementation header for wcstoull -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_WCSTOULL_H
#define LLVM_LIBC_SRC_WCHAR_WCSTOULL_H

#include "hdr/types/wint_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

unsigned long long wcstoull(const wchar_t *__restrict str,
                            wchar_t **__restrict str_end, int base);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCHAR_WCSTOULL_H
