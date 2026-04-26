//===-- Implementation header for wcsncat ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_WCSNCAT_H
#define LLVM_LIBC_SRC_WCHAR_WCSNCAT_H

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

wchar_t *wcsncat(wchar_t *__restrict s1, const wchar_t *__restrict s2,
                 size_t n);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCHAR_WCSNCAT_H
