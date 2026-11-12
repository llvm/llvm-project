// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_ITERATOR_TRAITS_H
#define _LIBCPP___CXX03___ITERATOR_ITERATOR_TRAITS_H

#include <__cxx03/__config>
#include <__cxx03/__fwd/pair.h>
#include <__cxx03/__type_traits/conditional.h>
#include <__cxx03/__type_traits/disjunction.h>
#include <__cxx03/__type_traits/is_convertible.h>
#include <__cxx03/__type_traits/is_object.h>
#include <__cxx03/__type_traits/is_primary_template.h>
#include <__cxx03/__type_traits/is_reference.h>
#include <__cxx03/__type_traits/is_valid_expansion.h>
#include <__cxx03/__type_traits/remove_const.h>
#include <__cxx03/__type_traits/remove_cv.h>
#include <__cxx03/__type_traits/remove_cvref.h>
#include <__cxx03/__type_traits/void_t.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter>
struct _LIBCPP_TEMPLATE_VIS iterator_traits;

struct _LIBCPP_TEMPLATE_VIS input_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS output_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS forward_iterator_tag : public input_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS bidirectional_iterator_tag : public forward_iterator_tag {};
struct _LIBCPP_TEMPLATE_VIS random_access_iterator_tag : public bidirectional_iterator_tag {};

template <class _Iter>
struct __iter_traits_cache {
  using type = _If< __is_primary_template<iterator_traits<_Iter> >::value, _Iter, iterator_traits<_Iter> >;
};
template <class _Iter>
using _ITER_TRAITS = typename __iter_traits_cache<_Iter>::type;

struct __iter_concept_concept_test {
  template <class _Iter>
  using _Apply = typename _ITER_TRAITS<_Iter>::iterator_concept;
};
struct __iter_concept_category_test {
  template <class _Iter>
  using _Apply = typename _ITER_TRAITS<_Iter>::iterator_category;
};
struct __iter_concept_random_fallback {
  template <class _Iter>
  using _Apply = __enable_if_t< __is_primary_template<iterator_traits<_Iter> >::value, random_access_iterator_tag >;
};

template <class _Iter, class _Tester>
struct __test_iter_concept : _IsValidExpansion<_Tester::template _Apply, _Iter>, _Tester {};

template <class _Iter>
struct __iter_concept_cache {
  using type = _Or< __test_iter_concept<_Iter, __iter_concept_concept_test>,
                    __test_iter_concept<_Iter, __iter_concept_category_test>,
                    __test_iter_concept<_Iter, __iter_concept_random_fallback> >;
};

template <class _Iter>
using _ITER_CONCEPT = typename __iter_concept_cache<_Iter>::type::template _Apply<_Iter>;

template <class _Tp>
struct __has_iterator_typedefs {
private:
  template <class _Up>
  static false_type __test(...);
  template <class _Up>
  static true_type
  __test(__void_t<typename _Up::iterator_category>* = nullptr,
         __void_t<typename _Up::difference_type>*   = nullptr,
         __void_t<typename _Up::value_type>*        = nullptr,
         __void_t<typename _Up::reference>*         = nullptr,
         __void_t<typename _Up::pointer>*           = nullptr);

public:
  static const bool value = decltype(__test<_Tp>(nullptr, nullptr, nullptr, nullptr, nullptr))::value;
};

template <class _Tp>
struct __has_iterator_category {
private:
  template <class _Up>
  static false_type __test(...);
  template <class _Up>
  static true_type __test(typename _Up::iterator_category* = nullptr);

public:
  static const bool value = decltype(__test<_Tp>(nullptr))::value;
};

template <class _Tp>
struct __has_iterator_concept {
private:
  template <class _Up>
  static false_type __test(...);
  template <class _Up>
  static true_type __test(typename _Up::iterator_concept* = nullptr);

public:
  static const bool value = decltype(__test<_Tp>(nullptr))::value;
};

template <class _Iter, bool>
struct __iterator_traits {};

template <class _Iter, bool>
struct __iterator_traits_impl {};

template <class _Iter>
struct __iterator_traits_impl<_Iter, true> {
  typedef typename _Iter::difference_type difference_type;
  typedef typename _Iter::value_type value_type;
  typedef typename _Iter::pointer pointer;
  typedef typename _Iter::reference reference;
  typedef typename _Iter::iterator_category iterator_category;
};

template <class _Iter>
struct __iterator_traits<_Iter, true>
    : __iterator_traits_impl< _Iter,
                              is_convertible<typename _Iter::iterator_category, input_iterator_tag>::value ||
                                  is_convertible<typename _Iter::iterator_category, output_iterator_tag>::value > {};

// iterator_traits<Iterator> will only have the nested types if Iterator::iterator_category
//    exists.  Else iterator_traits<Iterator> will be an empty class.  This is a
//    conforming extension which allows some programs to compile and behave as
//    the client expects instead of failing at compile time.

template <class _Iter>
struct _LIBCPP_TEMPLATE_VIS iterator_traits : __iterator_traits<_Iter, __has_iterator_typedefs<_Iter>::value> {
  using __primary_template = iterator_traits;
};

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS iterator_traits<_Tp*> {
  typedef ptrdiff_t difference_type;
  typedef __remove_cv_t<_Tp> value_type;
  typedef _Tp* pointer;
  typedef _Tp& reference;
  typedef random_access_iterator_tag iterator_category;
};

template <class _Tp, class _Up, bool = __has_iterator_category<iterator_traits<_Tp> >::value>
struct __has_iterator_category_convertible_to : is_convertible<typename iterator_traits<_Tp>::iterator_category, _Up> {
};

template <class _Tp, class _Up>
struct __has_iterator_category_convertible_to<_Tp, _Up, false> : false_type {};

template <class _Tp, class _Up, bool = __has_iterator_concept<_Tp>::value>
struct __has_iterator_concept_convertible_to : is_convertible<typename _Tp::iterator_concept, _Up> {};

template <class _Tp, class _Up>
struct __has_iterator_concept_convertible_to<_Tp, _Up, false> : false_type {};

template <class _Tp>
using __has_input_iterator_category = __has_iterator_category_convertible_to<_Tp, input_iterator_tag>;

template <class _Tp>
using __has_forward_iterator_category = __has_iterator_category_convertible_to<_Tp, forward_iterator_tag>;

template <class _Tp>
using __has_bidirectional_iterator_category = __has_iterator_category_convertible_to<_Tp, bidirectional_iterator_tag>;

template <class _Tp>
using __has_random_access_iterator_category = __has_iterator_category_convertible_to<_Tp, random_access_iterator_tag>;

// __libcpp_is_contiguous_iterator determines if an iterator is known by
// libc++ to be contiguous, either because it advertises itself as such
// (in C++20) or because it is a pointer type or a known trivial wrapper
// around a (possibly fancy) pointer type, such as __wrap_iter<T*>.
// Such iterators receive special "contiguous" optimizations in
// std::copy and std::sort.
//
template <class _Tp>
struct __libcpp_is_contiguous_iterator : false_type {};

// Any native pointer which is an iterator is also a contiguous iterator.
template <class _Up>
struct __libcpp_is_contiguous_iterator<_Up*> : true_type {};

template <class _Iter>
class __wrap_iter;

template <class _Tp>
using __has_exactly_input_iterator_category =
    integral_constant<bool,
                      __has_iterator_category_convertible_to<_Tp, input_iterator_tag>::value &&
                          !__has_iterator_category_convertible_to<_Tp, forward_iterator_tag>::value>;

template <class _Tp>
using __has_exactly_forward_iterator_category =
    integral_constant<bool,
                      __has_iterator_category_convertible_to<_Tp, forward_iterator_tag>::value &&
                          !__has_iterator_category_convertible_to<_Tp, bidirectional_iterator_tag>::value>;

template <class _Tp>
using __has_exactly_bidirectional_iterator_category =
    integral_constant<bool,
                      __has_iterator_category_convertible_to<_Tp, bidirectional_iterator_tag>::value &&
                          !__has_iterator_category_convertible_to<_Tp, random_access_iterator_tag>::value>;

template <class _InputIterator>
using __iter_value_type = typename iterator_traits<_InputIterator>::value_type;

template <class _InputIterator>
using __iter_key_type = __remove_const_t<typename iterator_traits<_InputIterator>::value_type::first_type>;

template <class _InputIterator>
using __iter_mapped_type = typename iterator_traits<_InputIterator>::value_type::second_type;

template <class _InputIterator>
using __iter_to_alloc_type =
    pair<const typename iterator_traits<_InputIterator>::value_type::first_type,
         typename iterator_traits<_InputIterator>::value_type::second_type>;

template <class _Iter>
using __iterator_category_type = typename iterator_traits<_Iter>::iterator_category;

template <class _Iter>
using __iterator_pointer_type = typename iterator_traits<_Iter>::pointer;

template <class _Iter>
using __iter_diff_t = typename iterator_traits<_Iter>::difference_type;

template <class _Iter>
using __iter_reference = typename iterator_traits<_Iter>::reference;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ITERATOR_ITERATOR_TRAITS_H
