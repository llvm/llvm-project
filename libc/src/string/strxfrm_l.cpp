//===-- Implementation of strxfrm_l ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strxfrm_l.h"
#include "src/string/strxfrm.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

// TODO: Add support for locales.
LLVM_LIBC_FUNCTION(size_t, strxfrm_l,
                   (char *__restrict dest, const char *__restrict src, size_t n,
                    locale_t)) {
  return strxfrm(dest, src, n);
}

} // namespace LIBC_NAMESPACE
