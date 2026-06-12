//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for wcstoumax.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_INTTYPES_WCSTOUMAX_H
#define LLVM_LIBC_SRC_INTTYPES_WCSTOUMAX_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

uintmax_t wcstoumax(const wchar_t *__restrict str, wchar_t **__restrict str_end,
                    int base);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_INTTYPES_WCSTOUMAX_H
