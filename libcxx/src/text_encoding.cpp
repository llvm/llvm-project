//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <text_encoding>

_LIBCPP_BEGIN_NAMESPACE_STD

__text_encoding_rep __get_locale_encoding(const char* __name) {
  __text_encoding_rep __encoding{};
  if (auto __loc = __locale::__newlocale(LC_CTYPE_MASK, __name, static_cast<locale_t>(0))) {
    if (const char* __codeset = __locale::__nl_langinfo_l(CODESET, __loc)) {
      string_view __s(__codeset);
      if (__s.size() <= __text_encoding_rep::__max_name_length_)
        __encoding = __text_encoding_rep(__s);
    }
    __locale::__freelocale(__loc);
  }
  return __encoding;
}

_LIBCPP_END_NAMESPACE_STD
