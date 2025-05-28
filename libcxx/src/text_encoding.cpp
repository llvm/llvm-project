//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>

#include <__locale_dir/locale_base_api.h>
#include <__text_encoding/text_encoding_get_locale.h>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

string_view __get_locale_encoding(const char* __name){
  std::string_view __encoding_str{""};
  if (auto __loc = __locale::__newlocale(LC_CTYPE_MASK, __name, static_cast<locale_t>(0))) {
    if (const char* __codeset = __locale::__nl_langinfo_l(CODESET, __loc)) {
      string_view __s(__codeset);
      if (__s.size() < 63)
        __encoding_str = __s; 
    }
    __locale::__freelocale(__loc);
  }
  return __encoding_str;
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
