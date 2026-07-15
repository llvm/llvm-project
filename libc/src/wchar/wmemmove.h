//===-- Implementation header for wmemmove --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_WMEMMOVE_H
#define LLVM_LIBC_SRC_WCHAR_WMEMMOVE_H

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

wchar_t *wmemmove(wchar_t *dest, const wchar_t *src, size_t n);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCHAR_WMEMMOVE_H
