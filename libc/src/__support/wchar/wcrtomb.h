//===-- Implementation header for wcrtomb ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC__SUPPORT_WCHAR_WCRTOMB_H
#define LLVM_LIBC_SRC__SUPPORT_WCHAR_WCRTOMB_H

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

LIBC_INLINE ErrorOr<size_t> wcrtomb(char *__restrict dest_ptr, wchar_t wc,
                        mbstate *__restrict ps) {
  LIBC_CRASH_ON_NULLPTR(ps);
  CharacterConverter char_conv(ps);
  return char_conv.rtomb_generic<wchar_t>(dest_ptr, wc);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC__SUPPORT_WCHAR_WCRTOMB_H
