//===-- Implementation header for wcsrtombs -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC__SUPPORT_WCHAR_WCSRTOMBS_H
#define LLVM_LIBC_SRC__SUPPORT_WCHAR_WCSRTOMBS_H

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<size_t> wcsrtombs(char *__restrict dst, const wchar_t **__restrict src,
                          size_t len, mbstate_t *__restrict ps);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC__SUPPORT_WCHAR_WCSRTOMBS_H
