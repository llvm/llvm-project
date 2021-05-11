// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_RAW_STORAGE_ITERATOR_H
#define _LIBCPP___MEMORY_RAW_STORAGE_ITERATOR_H

#include <__config>
#include <__memory/addressof.h>
#include <cstddef>
#include <iterator>
#include <utility>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER <= 17

template <class _OutputIterator, class _Tp>
class _LIBCPP_TEMPLATE_VIS _LIBCPP_DEPRECATED_IN_CXX17 raw_storage_iterator
    : public iterator<output_iterator_tag,
                      _Tp,                                         // purposefully not C++03
                      ptrdiff_t,                                   // purposefully not C++03
                      _Tp*,                                        // purposefully not C++03
                      raw_storage_iterator<_OutputIterator, _Tp>&> // purposefully not C++03
{
private:
    _OutputIterator __x_;
public:
    _LIBCPP_INLINE_VISIBILITY explicit raw_storage_iterator(_OutputIterator __x) : __x_(__x) {}
    _LIBCPP_INLINE_VISIBILITY raw_storage_iterator& operator*() {return *this;}
    _LIBCPP_INLINE_VISIBILITY raw_storage_iterator& operator=(const _Tp& __element)
        {::new ((void*)_VSTD::addressof(*__x_)) _Tp(__element); return *this;}
#if _LIBCPP_STD_VER >= 14
    _LIBCPP_INLINE_VISIBILITY raw_storage_iterator& operator=(_Tp&& __element)
        {::new ((void*)_VSTD::addressof(*__x_)) _Tp(_VSTD::move(__element)); return *this;}
#endif
    _LIBCPP_INLINE_VISIBILITY raw_storage_iterator& operator++() {++__x_; return *this;}
    _LIBCPP_INLINE_VISIBILITY raw_storage_iterator  operator++(int)
        {raw_storage_iterator __t(*this); ++__x_; return __t;}
#if _LIBCPP_STD_VER >= 14
    _LIBCPP_INLINE_VISIBILITY _OutputIterator base() const { return __x_; }
#endif
};

#endif // _LIBCPP_STD_VER <= 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_RAW_STORAGE_ITERATOR_H
