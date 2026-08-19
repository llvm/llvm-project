//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_COLLATE_H
#define _LIBCPP___LOCALE_DIR_COLLATE_H

#include <__config>

#if _LIBCPP_HAS_LOCALIZATION

#  include <__cstddef/size_t.h>
#  include <__locale_dir/locale.h>
#  include <__locale_dir/locale_base_api.h>
#  include <string>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

// template <class _CharT> class collate;

template <class _CharT>
class collate : public locale::facet {
public:
  typedef _CharT char_type;
  typedef basic_string<char_type> string_type;

  _LIBCPP_HIDE_FROM_ABI explicit collate(size_t __refs = 0) : locale::facet(__refs) {}

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int
  compare(const char_type* __lo1, const char_type* __hi1, const char_type* __lo2, const char_type* __hi2) const {
    return do_compare(__lo1, __hi1, __lo2, __hi2);
  }

  // FIXME(EricWF): The _LIBCPP_ALWAYS_INLINE is needed on Windows to work
  // around a dllimport bug that expects an external instantiation.
  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE string_type
  transform(const char_type* __lo, const char_type* __hi) const {
    return do_transform(__lo, __hi);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI long hash(const char_type* __lo, const char_type* __hi) const {
    return do_hash(__lo, __hi);
  }

  static locale::id id;

protected:
  ~collate() override;
  virtual int
  do_compare(const char_type* __lo1, const char_type* __hi1, const char_type* __lo2, const char_type* __hi2) const;
  virtual string_type do_transform(const char_type* __lo, const char_type* __hi) const {
    return string_type(__lo, __hi);
  }
  virtual long do_hash(const char_type* __lo, const char_type* __hi) const;
};

template <class _CharT>
locale::id collate<_CharT>::id;

template <class _CharT>
collate<_CharT>::~collate() {}

template <class _CharT>
int collate<_CharT>::do_compare(
    const char_type* __lo1, const char_type* __hi1, const char_type* __lo2, const char_type* __hi2) const {
  for (; __lo2 != __hi2; ++__lo1, ++__lo2) {
    if (__lo1 == __hi1 || *__lo1 < *__lo2)
      return -1;
    if (*__lo2 < *__lo1)
      return 1;
  }
  return __lo1 != __hi1;
}

template <class _CharT>
long collate<_CharT>::do_hash(const char_type* __lo, const char_type* __hi) const {
  size_t __h          = 0;
  const size_t __sr   = __CHAR_BIT__ * sizeof(size_t) - 8;
  const size_t __mask = size_t(0xF) << (__sr + 4);
  for (const char_type* __p = __lo; __p != __hi; ++__p) {
    __h        = (__h << 4) + static_cast<size_t>(*__p);
    size_t __g = __h & __mask;
    __h ^= __g | (__g >> __sr);
  }
  return static_cast<long>(__h);
}

extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS collate<char>;
#  if _LIBCPP_HAS_WIDE_CHARACTERS
extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS collate<wchar_t>;
#  endif

// template <class CharT> class collate_byname;

template <class _CharT>
class collate_byname;

template <>
class _LIBCPP_EXPORTED_FROM_ABI collate_byname<char> : public collate<char> {
  __locale::__locale_t __l_;

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

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI collate_byname<wchar_t> : public collate<wchar_t> {
  __locale::__locale_t __l_;

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
#  endif

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP___LOCALE_DIR_COLLATE_H
