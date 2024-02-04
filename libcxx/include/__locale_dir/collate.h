//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_COLLATE_H
#define _LIBCPP___LOCALE_COLLATE_H

#include <__config>
#include <__locale_dir/locale.h>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS collate : public locale::facet {
public:
  typedef _CharT char_type;
  typedef basic_string<char_type> string_type;

  _LIBCPP_HIDE_FROM_ABI explicit collate(size_t __refs = 0) : locale::facet(__refs) {}

  _LIBCPP_HIDE_FROM_ABI int
  compare(const char_type* __lo1, const char_type* __hi1, const char_type* __lo2, const char_type* __hi2) const {
    return do_compare(__lo1, __hi1, __lo2, __hi2);
  }

  // FIXME(EricWF): The _LIBCPP_ALWAYS_INLINE is needed on Windows to work
  // around a dllimport bug that expects an external instantiation.
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE string_type
  transform(const char_type* __lo, const char_type* __hi) const {
    return do_transform(__lo, __hi);
  }

  _LIBCPP_HIDE_FROM_ABI long hash(const char_type* __lo, const char_type* __hi) const { return do_hash(__lo, __hi); }

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
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS collate<wchar_t>;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_COLLATE_H
