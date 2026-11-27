//===-- Implementation of btowc -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/btowc.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h" // for WEOF.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wint_t, btowc, (int c)) {
  if (c > 127 || c < 0)
    return WEOF;
  return static_cast<wint_t>(c);
}

} // namespace LIBC_NAMESPACE_DECL
