//===-- String Converter for printf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_CONVERTER_H

#ifndef LIBC_COPT_PRINTF_DISABLE_WIDE
#include "hdr/types/wchar_t.h"
#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcrtomb.h"
#endif // LIBC_COPT_PRINTF_DISABLE_WIDE

#include "src/__support/CPP/algorithm.h" // cpp::min
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/converter_utils.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"
#include "src/string/string_length.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

template <WriteMode write_mode>
LIBC_INLINE int convert_string(Writer<write_mode> *writer,
                               const FormatSection &to_conv) {
  size_t string_len = 0;

  if (to_conv.length_modifier == LengthModifier::l) {
#ifndef LIBC_COPT_PRINTF_DISABLE_WIDE
    const wchar_t *wstr_ptr =
        reinterpret_cast<const wchar_t *>(to_conv.conv_val_ptr);

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    if (wstr_ptr == nullptr) {
      wstr_ptr = L"(null)";
    }
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS

    char buffer[MB_LEN_MAX];
    internal::mbstate mbstate;
    size_t written = 0;
    for (const wchar_t *cur_str = (wstr_ptr); cur_str[string_len];
         ++string_len) {
      wchar_t wc = cur_str[string_len];

      auto ret = internal::wcrtomb(buffer, wc, &mbstate);
      if (!ret.has_value()) {
        return MB_CONVERSION_ERROR;
      }
      written += ret.value();

      if (to_conv.precision >= 0 &&
          static_cast<size_t>(to_conv.precision) < written) {
        written -= ret.value();
        break;
      }
    }
    string_len = written;
#else

    const char *str_ptr = reinterpret_cast<const char *>(to_conv.conv_val_ptr);

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
  if (str_ptr == nullptr) {
    str_ptr = "(null)";
  }
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS

  string_len = cpp::min(internal::string_length(str_ptr),
                        static_cast<size_t>(to_conv.precision));

#endif // LIBC_COPT_PRINTF_DISABLE_WIDE
  } else {

    const char *str_ptr = reinterpret_cast<const char *>(to_conv.conv_val_ptr);
#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    if (str_ptr == nullptr) {
      str_ptr = "(null)";
    }
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    string_len = cpp::min(internal::string_length(str_ptr),
                          static_cast<size_t>(to_conv.precision));
  }

  size_t padding_spaces = to_conv.min_width > static_cast<int>(string_len)
                              ? to_conv.min_width - string_len
                              : 0;

  // If the padding is on the left side, write the spaces first.
  if (padding_spaces > 0 &&
      (to_conv.flags & FormatFlags::LEFT_JUSTIFIED) == 0) {
    RET_IF_RESULT_NEGATIVE(writer->write(' ', padding_spaces));
  }

  if (to_conv.length_modifier == LengthModifier::l) {
#ifndef LIBC_COPT_PRINTF_DISABLE_WIDE
    const wchar_t *wstr_ptr =
        reinterpret_cast<const wchar_t *>(to_conv.conv_val_ptr);

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    if (wstr_ptr == nullptr) {
      wstr_ptr = L"(null)";
    }
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS

    size_t written = 0;
    char buffer[MB_LEN_MAX];
    internal::mbstate mbstate;
    for (size_t i = 0; written < string_len; ++i) {
      // We don't need to check errors/precision here; Pass 1 guaranteed safety.
      auto ret = internal::wcrtomb(buffer, wstr_ptr[i], &mbstate);
      size_t mb_len = ret.value();

      RET_IF_RESULT_NEGATIVE(writer->write({buffer, mb_len}));
      written += mb_len;
    }
#else
    const char *str_ptr = reinterpret_cast<const char *>(to_conv.conv_val_ptr);

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    if (str_ptr == nullptr) {
      str_ptr = "(null)";
    }
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    RET_IF_RESULT_NEGATIVE(writer->write({(str_ptr), string_len}));
#endif // LIBC_COPT_PRINTF_DISABLE_WIDE
  } else {
    const char *str_ptr = reinterpret_cast<const char *>(to_conv.conv_val_ptr);

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    if (str_ptr == nullptr) {
      str_ptr = "(null)";
    }
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
    RET_IF_RESULT_NEGATIVE(writer->write({(str_ptr), string_len}));
  }

  // If the padding is on the right side, write the spaces last.
  if (padding_spaces > 0 &&
      (to_conv.flags & FormatFlags::LEFT_JUSTIFIED) != 0) {
    RET_IF_RESULT_NEGATIVE(writer->write(' ', padding_spaces));
  }
  return WRITE_OK;
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_STRING_CONVERTER_H
