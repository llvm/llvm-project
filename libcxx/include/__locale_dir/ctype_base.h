//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_CTYPE_BASE_H
#define _LIBCPP___LOCALE_DIR_CTYPE_BASE_H

#include <__config>

#if _LIBCPP_HAS_LOCALIZATION

#  include <__configuration/platform.h>
#  include <__locale_dir/locale_base_api.h>
#  include <__type_traits/make_unsigned.h>
#  include <cctype>
#  include <cstdint>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

class _LIBCPP_EXPORTED_FROM_ABI ctype_base : public __locale::__ctype_base {
public:
  static const mask alnum = alpha | digit;
  static const mask graph = alnum | punct;

  _LIBCPP_HIDE_FROM_ABI ctype_base() {}

  static_assert((__regex_word & ~(std::make_unsigned<mask>::type)(space | print | cntrl | upper | lower | alpha |
                                                                  digit | punct | xdigit | blank)) == __regex_word,
                "__regex_word can't overlap other bits");
};

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP___LOCALE_DIR_CTYPE_BASE_H
