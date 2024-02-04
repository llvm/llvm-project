//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_COLLATE_BYNAME_H
#define _LIBCPP___LOCALE_COLLATE_BYNAME_H

#include <__config>
#include <__locale_dir/collate.h>
#include <__locale_dir/locale_base_api.h>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS collate_byname;

template <>
class _LIBCPP_EXPORTED_FROM_ABI collate_byname<char> : public collate<char> {
  locale_t __l_;

public:
  typedef char char_type;
  typedef basic_string<char_type> string_type;

  explicit collate_byname(const char* __n, size_t __refs = 0);
  explicit collate_byname(const string& __n, size_t __refs = 0);

protected:
  ~collate_byname() override;
  int do_compare(
      const char_type* __lo1, const char_type* __hi1, const char_type* __lo2, const char_type* __hi2) const override;
  string_type do_transform(const char_type* __lo, const char_type* __hi) const override;
};

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI collate_byname<wchar_t> : public collate<wchar_t> {
  locale_t __l_;

public:
  typedef wchar_t char_type;
  typedef basic_string<char_type> string_type;

  explicit collate_byname(const char* __n, size_t __refs = 0);
  explicit collate_byname(const string& __n, size_t __refs = 0);

protected:
  ~collate_byname() override;

  int do_compare(
      const char_type* __lo1, const char_type* __hi1, const char_type* __lo2, const char_type* __hi2) const override;
  string_type do_transform(const char_type* __lo, const char_type* __hi) const override;
};
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_COLLATE_BYNAME_H
