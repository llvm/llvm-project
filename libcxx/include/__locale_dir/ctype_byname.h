//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_CTYPE_BYNAME_H
#define _LIBCPP___LOCALE_CTYPE_BYNAME_H

#include <__config>
#include <__locale_dir/ctype.h>
#include <__locale_dir/ctype_base.h>
#include <__locale_dir/locale_base_api.h>
#include <cstddef>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS ctype_byname;

template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype_byname<char> : public ctype<char> {
  locale_t __l_;

public:
  explicit ctype_byname(const char*, size_t = 0);
  explicit ctype_byname(const string&, size_t = 0);

protected:
  ~ctype_byname() override;
  char_type do_toupper(char_type) const override;
  const char_type* do_toupper(char_type* __low, const char_type* __high) const override;
  char_type do_tolower(char_type) const override;
  const char_type* do_tolower(char_type* __low, const char_type* __high) const override;
};

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype_byname<wchar_t> : public ctype<wchar_t> {
  locale_t __l_;

public:
  explicit ctype_byname(const char*, size_t = 0);
  explicit ctype_byname(const string&, size_t = 0);

protected:
  ~ctype_byname() override;
  bool do_is(mask __m, char_type __c) const override;
  const char_type* do_is(const char_type* __low, const char_type* __high, mask* __vec) const override;
  const char_type* do_scan_is(mask __m, const char_type* __low, const char_type* __high) const override;
  const char_type* do_scan_not(mask __m, const char_type* __low, const char_type* __high) const override;
  char_type do_toupper(char_type) const override;
  const char_type* do_toupper(char_type* __low, const char_type* __high) const override;
  char_type do_tolower(char_type) const override;
  const char_type* do_tolower(char_type* __low, const char_type* __high) const override;
  char_type do_widen(char) const override;
  const char* do_widen(const char* __low, const char* __high, char_type* __dest) const override;
  char do_narrow(char_type, char __dfault) const override;
  const char_type*
  do_narrow(const char_type* __low, const char_type* __high, char __dfault, char* __dest) const override;
};
#endif // _LIBCPP_HAS_NO_WIDE_CHARACTERS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_CTYPE_BYNAME_H
