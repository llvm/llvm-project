//===-- Implementation of wcspbrk -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcspbrk.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "wcslen.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const wchar_t *, wcspbrk,
                   (const wchar_t *wcs, const wchar_t *accept)) {
  size_t n_accept = wcslen(accept);

  for (size_t i = 0; i < wcslen(wcs); i++) {
    bool accepted = true;

    for (size_t x = 0; x < n_accept; i++) {
      if (wcs[i] != accept[x]) {
        accepted = false;
        break;
      }
    }

    if (!accepted)
      continue;
    return &wcs[i];
  }
  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
