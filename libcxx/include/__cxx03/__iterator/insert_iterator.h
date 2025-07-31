// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_INSERT_ITERATOR_H
#define _LIBCPP___CXX03___ITERATOR_INSERT_ITERATOR_H

#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Container>
using __insert_iterator_iter_t = typename _Container::iterator;

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _Container>
class _LIBCPP_TEMPLATE_VIS insert_iterator : public iterator<output_iterator_tag, void, void, void, void> {
  _LIBCPP_SUPPRESS_DEPRECATED_POP

protected:
  _Container* container;
  __insert_iterator_iter_t<_Container> iter;

public:
  typedef output_iterator_tag iterator_category;
  typedef void value_type;
  typedef void difference_type;
  typedef void pointer;
  typedef void reference;
  typedef _Container container_type;

  _LIBCPP_HIDE_FROM_ABI insert_iterator(_Container& __x, __insert_iterator_iter_t<_Container> __i)
      : container(std::addressof(__x)), iter(__i) {}
  _LIBCPP_HIDE_FROM_ABI insert_iterator& operator=(const typename _Container::value_type& __value) {
    iter = container->insert(iter, __value);
    ++iter;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI insert_iterator& operator*() { return *this; }
  _LIBCPP_HIDE_FROM_ABI insert_iterator& operator++() { return *this; }
  _LIBCPP_HIDE_FROM_ABI insert_iterator& operator++(int) { return *this; }
};

template <class _Container>
inline _LIBCPP_HIDE_FROM_ABI insert_iterator<_Container>
inserter(_Container& __x, __insert_iterator_iter_t<_Container> __i) {
  return insert_iterator<_Container>(__x, __i);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ITERATOR_INSERT_ITERATOR_H
