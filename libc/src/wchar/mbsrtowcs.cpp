//===-- Implementation of mbsrtowcs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/mbsrtowcs.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/string_converter.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, mbsrtowcs,
                   (wchar_t *__restrict dst, const char **__restrict src,
                    size_t len, mbstate_t *__restrict ps)) {
  static internal::mbstate internal_mbstate;
  internal::StringConverter<char8_t> str_conv(
      reinterpret_cast<const char8_t *>(src),
      ps == nullptr ? &internal_mbstate
                    : reinterpret_cast<internal::mbstate *>(ps),
      len);

  int dst_idx = 0;
  ErrorOr<char32_t> converted = str_conv.popUTF32();
  while (converted.has_value()) {
    dst[dst_idx] = converted.value();
    dst_idx++;
    converted = str_conv.popUTF32();
  }

  src += str_conv.getSourceIndex();
  if (converted.error() == -1) // if we hit conversion limit
    return dst_idx;

  libc_errno = converted.error();
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
