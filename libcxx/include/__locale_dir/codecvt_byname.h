//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_CODECVT_BYNAME_H
#define _LIBCPP___LOCALE_CODECVT_BYNAME_H

#include <__config>
#include <__locale_dir/codecvt.h>
#include <cstddef>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InternT, class _ExternT, class _StateT>
class _LIBCPP_TEMPLATE_VIS codecvt_byname : public codecvt<_InternT, _ExternT, _StateT> {
public:
  _LIBCPP_HIDE_FROM_ABI explicit codecvt_byname(const char* __nm, size_t __refs = 0)
      : codecvt<_InternT, _ExternT, _StateT>(__nm, __refs) {}
  _LIBCPP_HIDE_FROM_ABI explicit codecvt_byname(const string& __nm, size_t __refs = 0)
      : codecvt<_InternT, _ExternT, _StateT>(__nm.c_str(), __refs) {}

protected:
  ~codecvt_byname() override;
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _InternT, class _ExternT, class _StateT>
codecvt_byname<_InternT, _ExternT, _StateT>::~codecvt_byname() {}
_LIBCPP_SUPPRESS_DEPRECATED_POP

extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char, char, mbstate_t>;
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<wchar_t, char, mbstate_t>;
#endif
extern template class _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS
    codecvt_byname<char16_t, char, mbstate_t>; // deprecated in C++20
extern template class _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS
    codecvt_byname<char32_t, char, mbstate_t>; // deprecated in C++20
#ifndef _LIBCPP_HAS_NO_CHAR8_T
extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char16_t, char8_t, mbstate_t>; // C++20
extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char32_t, char8_t, mbstate_t>; // C++20
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_CODECVT_BYNAME_H
