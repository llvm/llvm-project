//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_UNINITIALIZED_MULTIDIMENSIONAL_ALGORITHMS_H
#define _LIBCPP___MEMORY_UNINITIALIZED_MULTIDIMENSIONAL_ALGORITHMS_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__memory/allocator_traits.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/extent.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_extent.h>
#include <__utility/exception_guard.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

// TODO: Rewrite this to iterate left to right and use reverse_iterators when calling
// Destroys every element in the range [first, last) FROM RIGHT TO LEFT using allocator
// destruction. If elements are themselves C-style arrays, they are recursively destroyed
// in the same manner.
//
// This function assumes that destructors do not throw, and that the allocator is bound to
// the correct type.
template <class _Alloc,
          class _BidirIter,
          __enable_if_t<__has_bidirectional_iterator_category<_BidirIter>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr void
__allocator_destroy_multidimensional(_Alloc& __alloc, _BidirIter __first, _BidirIter __last) noexcept {
  using _ValueType = typename iterator_traits<_BidirIter>::value_type;
  static_assert(is_same_v<typename allocator_traits<_Alloc>::value_type, _ValueType>,
                "The allocator should already be rebound to the correct type");

  if (__first == __last)
    return;

  if constexpr (is_array_v<_ValueType>) {
    static_assert(!__is_unbounded_array_v<_ValueType>,
                  "arrays of unbounded arrays don't exist, but if they did we would mess up here");

    using _Element = remove_extent_t<_ValueType>;
    __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
    do {
      --__last;
      decltype(auto) __array = *__last;
      std::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + extent_v<_ValueType>);
    } while (__last != __first);
  } else {
    do {
      --__last;
      allocator_traits<_Alloc>::destroy(__alloc, std::addressof(*__last));
    } while (__last != __first);
  }
}

// Constructs the object at the given location using the allocator's construct method.
//
// If the object being constructed is an array, each element of the array is allocator-constructed,
// recursively. If an exception is thrown during the construction of an array, the initialized
// elements are destroyed in reverse order of initialization using allocator destruction.
//
// This function assumes that the allocator is bound to the correct type.
template <class _Alloc, class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr void __allocator_construct_at_multidimensional(_Alloc& __alloc, _Tp* __loc) {
  static_assert(is_same_v<typename allocator_traits<_Alloc>::value_type, _Tp>,
                "The allocator should already be rebound to the correct type");

  if constexpr (is_array_v<_Tp>) {
    using _Element = remove_extent_t<_Tp>;
    __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
    size_t __i   = 0;
    _Tp& __array = *__loc;

    // If an exception is thrown, destroy what we have constructed so far in reverse order.
    auto __guard = std::__make_exception_guard([&]() {
      std::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + __i);
    });

    for (; __i != extent_v<_Tp>; ++__i) {
      std::__allocator_construct_at_multidimensional(__elem_alloc, std::addressof(__array[__i]));
    }
    __guard.__complete();
  } else {
    allocator_traits<_Alloc>::construct(__alloc, __loc);
  }
}

// Constructs the object at the given location using the allocator's construct method, passing along
// the provided argument.
//
// If the object being constructed is an array, the argument is also assumed to be an array. Each
// each element of the array being constructed is allocator-constructed from the corresponding
// element of the argument array. If an exception is thrown during the construction of an array,
// the initialized elements are destroyed in reverse order of initialization using allocator
// destruction.
//
// This function assumes that the allocator is bound to the correct type.
template <class _Alloc, class _Tp, class _Arg>
_LIBCPP_HIDE_FROM_ABI constexpr void
__allocator_construct_at_multidimensional(_Alloc& __alloc, _Tp* __loc, _Arg const& __arg) {
  static_assert(is_same_v<typename allocator_traits<_Alloc>::value_type, _Tp>,
                "The allocator should already be rebound to the correct type");

  if constexpr (is_array_v<_Tp>) {
    static_assert(is_array_v<_Arg>,
                  "Provided non-array initialization argument to __allocator_construct_at_multidimensional when "
                  "trying to construct an array.");

    using _Element = remove_extent_t<_Tp>;
    __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
    size_t __i   = 0;
    _Tp& __array = *__loc;

    // If an exception is thrown, destroy what we have constructed so far in reverse order.
    auto __guard = std::__make_exception_guard([&]() {
      std::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + __i);
    });
    for (; __i != extent_v<_Tp>; ++__i) {
      std::__allocator_construct_at_multidimensional(__elem_alloc, std::addressof(__array[__i]), __arg[__i]);
    }
    __guard.__complete();
  } else {
    allocator_traits<_Alloc>::construct(__alloc, __loc, __arg);
  }
}

// Given a range starting at it and containing n elements, initializes each element in the
// range from left to right using the construct method of the allocator (rebound to the
// correct type).
//
// If an exception is thrown, the initialized elements are destroyed in reverse order of
// initialization using allocator_traits destruction. If the elements in the range are C-style
// arrays, they are initialized element-wise using allocator construction, and recursively so.
template <class _Alloc,
          class _BidirIter,
          class _Tp,
          class _Size = typename iterator_traits<_BidirIter>::difference_type>
_LIBCPP_HIDE_FROM_ABI constexpr void
__uninitialized_allocator_fill_n_multidimensional(_Alloc& __alloc, _BidirIter __it, _Size __n, _Tp const& __value) {
  using _ValueType = typename iterator_traits<_BidirIter>::value_type;
  __allocator_traits_rebind_t<_Alloc, _ValueType> __value_alloc(__alloc);
  _BidirIter __begin = __it;

  // If an exception is thrown, destroy what we have constructed so far in reverse order.
  auto __guard =
      std::__make_exception_guard([&]() { std::__allocator_destroy_multidimensional(__value_alloc, __begin, __it); });
  for (; __n != 0; --__n, ++__it) {
    std::__allocator_construct_at_multidimensional(__value_alloc, std::addressof(*__it), __value);
  }
  __guard.__complete();
}

// Same as __uninitialized_allocator_fill_n_multidimensional, but doesn't pass any initialization argument
// to the allocator's construct method, which results in value initialization.
template <class _Alloc, class _BidirIter, class _Size = typename iterator_traits<_BidirIter>::difference_type>
_LIBCPP_HIDE_FROM_ABI constexpr void
__uninitialized_allocator_value_construct_n_multidimensional(_Alloc& __alloc, _BidirIter __it, _Size __n) {
  using _ValueType = typename iterator_traits<_BidirIter>::value_type;
  __allocator_traits_rebind_t<_Alloc, _ValueType> __value_alloc(__alloc);
  _BidirIter __begin = __it;

  // If an exception is thrown, destroy what we have constructed so far in reverse order.
  auto __guard =
      std::__make_exception_guard([&]() { std::__allocator_destroy_multidimensional(__value_alloc, __begin, __it); });
  for (; __n != 0; --__n, ++__it) {
    std::__allocator_construct_at_multidimensional(__value_alloc, std::addressof(*__it));
  }
  __guard.__complete();
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_UNINITIALIZED_MULTIDIMENSIONAL_ALGORITHMS_H
