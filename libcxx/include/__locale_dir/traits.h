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
#include <__locale_dir/locale.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isspace(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::space, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isprint(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::print, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool iscntrl(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::cntrl, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isupper(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::upper, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool islower(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::lower, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isalpha(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::alpha, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isdigit(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::digit, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool ispunct(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::punct, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isxdigit(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::xdigit, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isalnum(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::alnum, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI bool isgraph(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::graph, __c);
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI bool isblank(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::blank, __c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI _CharT toupper(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).toupper(__c);
}

template <class _CharT>
inline _LIBCPP_HIDE_FROM_ABI _CharT tolower(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).tolower(__c);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_CTYPE_BYNAME_H
