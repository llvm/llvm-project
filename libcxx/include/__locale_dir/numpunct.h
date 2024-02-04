//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_NUMPUNCT_H
#define _LIBCPP___LOCALE_NUMPUNCT_H

#include <__config>
#include <__locale_dir/locale.h>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS numpunct;

template <>
class _LIBCPP_EXPORTED_FROM_ABI numpunct<char> : public locale::facet {
public:
  typedef char char_type;
  typedef basic_string<char_type> string_type;

  explicit numpunct(size_t __refs = 0);

  _LIBCPP_HIDE_FROM_ABI char_type decimal_point() const { return do_decimal_point(); }
  _LIBCPP_HIDE_FROM_ABI char_type thousands_sep() const { return do_thousands_sep(); }
  _LIBCPP_HIDE_FROM_ABI string grouping() const { return do_grouping(); }
  _LIBCPP_HIDE_FROM_ABI string_type truename() const { return do_truename(); }
  _LIBCPP_HIDE_FROM_ABI string_type falsename() const { return do_falsename(); }

  static locale::id id;

protected:
  ~numpunct() override;
  virtual char_type do_decimal_point() const;
  virtual char_type do_thousands_sep() const;
  virtual string do_grouping() const;
  virtual string_type do_truename() const;
  virtual string_type do_falsename() const;

  char_type __decimal_point_;
  char_type __thousands_sep_;
  string __grouping_;
};

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI numpunct<wchar_t> : public locale::facet {
public:
  typedef wchar_t char_type;
  typedef basic_string<char_type> string_type;

  explicit numpunct(size_t __refs = 0);

  _LIBCPP_HIDE_FROM_ABI char_type decimal_point() const { return do_decimal_point(); }
  _LIBCPP_HIDE_FROM_ABI char_type thousands_sep() const { return do_thousands_sep(); }
  _LIBCPP_HIDE_FROM_ABI string grouping() const { return do_grouping(); }
  _LIBCPP_HIDE_FROM_ABI string_type truename() const { return do_truename(); }
  _LIBCPP_HIDE_FROM_ABI string_type falsename() const { return do_falsename(); }

  static locale::id id;

protected:
  ~numpunct() override;
  virtual char_type do_decimal_point() const;
  virtual char_type do_thousands_sep() const;
  virtual string do_grouping() const;
  virtual string_type do_truename() const;
  virtual string_type do_falsename() const;

  char_type __decimal_point_;
  char_type __thousands_sep_;
  string __grouping_;
};
#endif // _LIBCPP_HAS_NO_WIDE_CHARACTERS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_NUMPUNCT_H
