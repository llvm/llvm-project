//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__locale_dir/locale_base_api.h>
#include <__utility/scope_guard.h>
#include <text_encoding>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

static text_encoding __make_text_encoding(const char* __name) {
  if (__name == nullptr)
    return text_encoding{};

  string_view __name_view(__name);
  if (__name_view.size() > text_encoding::max_name_length)
    return text_encoding{};

  return text_encoding(__name_view);
}

#if defined(__ANDROID__)
// UTF-8 is the always the environment encoding on Android.
std::text_encoding __get_locale_encoding([[maybe_unused]] const char* __name) { return std::text_encoding::id::UTF8; }
#else
std::text_encoding __get_locale_encoding(const char* __name) {
  if (__name == nullptr)
    return __make_text_encoding(__locale::__get_locale_encoding(static_cast<__locale::__locale_t>(nullptr)));

  __locale::__locale_t __l = __locale::__newlocale(_LIBCPP_CTYPE_MASK, __name, static_cast<__locale::__locale_t>(0));

  __scope_guard __locale_guard([&__l] {
    if (__l) {
      __locale::__freelocale(__l);
    }
  });

  if (!__l) {
    return text_encoding{};
  }

  return __make_text_encoding(__locale::__get_locale_encoding(__l));
}

#endif // __ANDROID__

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
