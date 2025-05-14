// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___MEMORY_RAW_STORAGE_ITERATOR_H
#define _LIBCPP___CXX03___MEMORY_RAW_STORAGE_ITERATOR_H

#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/cstddef>
#include <__cxx03/new>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _OutputIterator, class _Tp>
class _LIBCPP_TEMPLATE_VIS raw_storage_iterator : public iterator<output_iterator_tag, void, void, void, void> {
  _LIBCPP_SUPPRESS_DEPRECATED_POP

private:
  _OutputIterator __x_;

public:
  typedef output_iterator_tag iterator_category;
  typedef void value_type;
  typedef void difference_type;
  typedef void pointer;
  typedef void reference;

  _LIBCPP_HIDE_FROM_ABI explicit raw_storage_iterator(_OutputIterator __x) : __x_(__x) {}
  _LIBCPP_HIDE_FROM_ABI raw_storage_iterator& operator*() { return *this; }
  _LIBCPP_HIDE_FROM_ABI raw_storage_iterator& operator=(const _Tp& __element) {
    ::new ((void*)std::addressof(*__x_)) _Tp(__element);
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI raw_storage_iterator& operator++() {
    ++__x_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI raw_storage_iterator operator++(int) {
    raw_storage_iterator __t(*this);
    ++__x_;
    return __t;
  }
};

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___MEMORY_RAW_STORAGE_ITERATOR_H
