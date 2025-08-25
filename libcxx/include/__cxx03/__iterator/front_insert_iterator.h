// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_FRONT_INSERT_ITERATOR_H
#define _LIBCPP___CXX03___ITERATOR_FRONT_INSERT_ITERATOR_H

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

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _Container>
class _LIBCPP_TEMPLATE_VIS front_insert_iterator : public iterator<output_iterator_tag, void, void, void, void> {
  _LIBCPP_SUPPRESS_DEPRECATED_POP

protected:
  _Container* container;

public:
  typedef output_iterator_tag iterator_category;
  typedef void value_type;
  typedef void difference_type;
  typedef void pointer;
  typedef void reference;
  typedef _Container container_type;

  _LIBCPP_HIDE_FROM_ABI explicit front_insert_iterator(_Container& __x) : container(std::addressof(__x)) {}
  _LIBCPP_HIDE_FROM_ABI front_insert_iterator& operator=(const typename _Container::value_type& __value) {
    container->push_front(__value);
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI front_insert_iterator& operator*() { return *this; }
  _LIBCPP_HIDE_FROM_ABI front_insert_iterator& operator++() { return *this; }
  _LIBCPP_HIDE_FROM_ABI front_insert_iterator operator++(int) { return *this; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(front_insert_iterator);

template <class _Container>
inline _LIBCPP_HIDE_FROM_ABI front_insert_iterator<_Container> front_inserter(_Container& __x) {
  return front_insert_iterator<_Container>(__x);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ITERATOR_FRONT_INSERT_ITERATOR_H
