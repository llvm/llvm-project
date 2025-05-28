//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#include <__locale_dir/locale_base_api.h>
#include <text_encoding>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

text_encoding text_encoding::environment() {
  auto __make_locale = [](const char* __name) {
    text_encoding __enc{};
    if (auto __loc = __locale::__newlocale(LC_CTYPE_MASK, __name, static_cast<locale_t>(0))) {
      if (const char* __codeset = __locale::__nl_langinfo_l(CODESET, __loc)) {
        string_view __s(__codeset);
        if (__s.size() < max_name_length)
          __enc = text_encoding(__s);
      }
      __locale::__freelocale(__loc);
    }
    return __enc;
  };
  return __make_locale("");
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
