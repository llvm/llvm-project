//===-- Implementation of wcsrtombs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsrtombs.h"

#include "hdr/types/char32_t.h"
#include "hdr/types/mbstate_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/string_converter.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, wcsrtombs,
                   (char *__restrict s, const wchar_t **__restrict pwcs,
                    size_t n, mbstate_t *ps)) {
  static internal::mbstate internal_mbstate;
  internal::StringConverter<char32_t> str_conv(
      reinterpret_cast<const char32_t *>(*pwcs),
      ps == nullptr ? &internal_mbstate
                    : reinterpret_cast<internal::mbstate *>(ps),
      n);

  int dst_idx = 0;
  ErrorOr<char8_t> converted = str_conv.popUTF8();
  while (converted.has_value()) {
    if (s != nullptr) 
      s[dst_idx] = converted.value();
    dst_idx++;
    converted = str_conv.popUTF8();
  }
  
  pwcs += str_conv.getSourceIndex();
  if (converted.error() == -1) // if we hit conversion limit
    return dst_idx;

  libc_errno = converted.error();
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
