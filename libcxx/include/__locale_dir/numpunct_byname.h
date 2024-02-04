//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_NUMPUNCT_BYNAME_H
#define _LIBCPP___LOCALE_NUMPUNCT_BYNAME_H

#include <__config>
#include <__locale_dir/locale.h>
#include <__locale_dir/numpunct.h>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS numpunct_byname;

template <>
class _LIBCPP_EXPORTED_FROM_ABI numpunct_byname<char> : public numpunct<char> {
public:
  typedef char char_type;
  typedef basic_string<char_type> string_type;

  explicit numpunct_byname(const char* __nm, size_t __refs = 0);
  explicit numpunct_byname(const string& __nm, size_t __refs = 0);

protected:
  ~numpunct_byname() override;

private:
  void __init(const char*);
};

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI numpunct_byname<wchar_t> : public numpunct<wchar_t> {
public:
  typedef wchar_t char_type;
  typedef basic_string<char_type> string_type;

  explicit numpunct_byname(const char* __nm, size_t __refs = 0);
  explicit numpunct_byname(const string& __nm, size_t __refs = 0);

protected:
  ~numpunct_byname() override;

private:
  void __init(const char*);
};
#endif // _LIBCPP_HAS_NO_WIDE_CHARACTERS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_NUMPUNCT_BYNAME_H
