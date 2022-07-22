// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_UNINITIALIZED_ALGORITHMS_H
#define _LIBCPP___MEMORY_UNINITIALIZED_ALGORITHMS_H

#include <__config>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__memory/allocator_traits.h>
#include <__memory/construct_at.h>
#include <__memory/voidify.h>
#include <__utility/move.h>
#include <__utility/pair.h>
#include <__utility/transaction.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// This is a simplified version of C++20 `unreachable_sentinel` that doesn't use concepts and thus can be used in any
// language mode.
struct __unreachable_sentinel {
  template <class _Iter>
  _LIBCPP_HIDE_FROM_ABI friend _LIBCPP_CONSTEXPR bool operator!=(const _Iter&, __unreachable_sentinel) _NOEXCEPT {
    return true;
  }
};

// uninitialized_copy

template <class _ValueType, class _InputIterator, class _Sentinel1, class _ForwardIterator, class _Sentinel2>
inline _LIBCPP_HIDE_FROM_ABI pair<_InputIterator, _ForwardIterator>
__uninitialized_copy(_InputIterator __ifirst, _Sentinel1 __ilast,
                     _ForwardIterator __ofirst, _Sentinel2 __olast) {
  _ForwardIterator __idx = __ofirst;
#ifndef _LIBCPP_NO_EXCEPTIONS
  try {
#endif
    for (; __ifirst != __ilast && __idx != __olast; ++__ifirst, (void)++__idx)
      ::new (_VSTD::__voidify(*__idx)) _ValueType(*__ifirst);
#ifndef _LIBCPP_NO_EXCEPTIONS
  } catch (...) {
    _VSTD::__destroy(__ofirst, __idx);
    throw;
  }
#endif

  return pair<_InputIterator, _ForwardIterator>(_VSTD::move(__ifirst), _VSTD::move(__idx));
}

template <class _InputIterator, class _ForwardIterator>
_ForwardIterator uninitialized_copy(_InputIterator __ifirst, _InputIterator __ilast,
                                    _ForwardIterator __ofirst) {
  typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
  auto __result = _VSTD::__uninitialized_copy<_ValueType>(_VSTD::move(__ifirst), _VSTD::move(__ilast),
                                                          _VSTD::move(__ofirst), __unreachable_sentinel());
  return _VSTD::move(__result.second);
}

// uninitialized_copy_n

template <class _ValueType, class _InputIterator, class _Size, class _ForwardIterator, class _Sentinel>
inline _LIBCPP_HIDE_FROM_ABI pair<_InputIterator, _ForwardIterator>
__uninitialized_copy_n(_InputIterator __ifirst, _Size __n,
                       _ForwardIterator __ofirst, _Sentinel __olast) {
  _ForwardIterator __idx = __ofirst;
#ifndef _LIBCPP_NO_EXCEPTIONS
  try {
#endif
    for (; __n > 0 && __idx != __olast; ++__ifirst, (void)++__idx, (void)--__n)
      ::new (_VSTD::__voidify(*__idx)) _ValueType(*__ifirst);
#ifndef _LIBCPP_NO_EXCEPTIONS
  } catch (...) {
    _VSTD::__destroy(__ofirst, __idx);
    throw;
  }
#endif

  return pair<_InputIterator, _ForwardIterator>(_VSTD::move(__ifirst), _VSTD::move(__idx));
}

template <class _InputIterator, class _Size, class _ForwardIterator>
inline _LIBCPP_HIDE_FROM_ABI _ForwardIterator uninitialized_copy_n(_InputIterator __ifirst, _Size __n,
                                                                   _ForwardIterator __ofirst) {
  typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
  auto __result = _VSTD::__uninitialized_copy_n<_ValueType>(_VSTD::move(__ifirst), __n, _VSTD::move(__ofirst),
                                                            __unreachable_sentinel());
  return _VSTD::move(__result.second);
}

// uninitialized_fill

template <class _ValueType, class _ForwardIterator, class _Sentinel, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator __uninitialized_fill(_ForwardIterator __first, _Sentinel __last, const _Tp& __x)
{
    _ForwardIterator __idx = __first;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif
        for (; __idx != __last; ++__idx)
            ::new (_VSTD::__voidify(*__idx)) _ValueType(__x);
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
        _VSTD::__destroy(__first, __idx);
        throw;
    }
#endif

    return __idx;
}

template <class _ForwardIterator, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI
void uninitialized_fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __x)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    (void)_VSTD::__uninitialized_fill<_ValueType>(__first, __last, __x);
}

// uninitialized_fill_n

template <class _ValueType, class _ForwardIterator, class _Size, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator __uninitialized_fill_n(_ForwardIterator __first, _Size __n, const _Tp& __x)
{
    _ForwardIterator __idx = __first;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try
    {
#endif
        for (; __n > 0; ++__idx, (void) --__n)
            ::new (_VSTD::__voidify(*__idx)) _ValueType(__x);
#ifndef _LIBCPP_NO_EXCEPTIONS
    }
    catch (...)
    {
        _VSTD::__destroy(__first, __idx);
        throw;
    }
#endif

    return __idx;
}

template <class _ForwardIterator, class _Size, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator uninitialized_fill_n(_ForwardIterator __first, _Size __n, const _Tp& __x)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    return _VSTD::__uninitialized_fill_n<_ValueType>(__first, __n, __x);
}

#if _LIBCPP_STD_VER > 14

// uninitialized_default_construct

template <class _ValueType, class _ForwardIterator, class _Sentinel>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator __uninitialized_default_construct(_ForwardIterator __first, _Sentinel __last) {
    auto __idx = __first;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try {
#endif
    for (; __idx != __last; ++__idx)
        ::new (_VSTD::__voidify(*__idx)) _ValueType;
#ifndef _LIBCPP_NO_EXCEPTIONS
    } catch (...) {
        _VSTD::__destroy(__first, __idx);
        throw;
    }
#endif

    return __idx;
}

template <class _ForwardIterator>
inline _LIBCPP_HIDE_FROM_ABI
void uninitialized_default_construct(_ForwardIterator __first, _ForwardIterator __last) {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    (void)_VSTD::__uninitialized_default_construct<_ValueType>(
        _VSTD::move(__first), _VSTD::move(__last));
}

// uninitialized_default_construct_n

template <class _ValueType, class _ForwardIterator, class _Size>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator __uninitialized_default_construct_n(_ForwardIterator __first, _Size __n) {
    auto __idx = __first;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try {
#endif
    for (; __n > 0; ++__idx, (void) --__n)
        ::new (_VSTD::__voidify(*__idx)) _ValueType;
#ifndef _LIBCPP_NO_EXCEPTIONS
    } catch (...) {
        _VSTD::__destroy(__first, __idx);
        throw;
    }
#endif

    return __idx;
}

template <class _ForwardIterator, class _Size>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator uninitialized_default_construct_n(_ForwardIterator __first, _Size __n) {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    return _VSTD::__uninitialized_default_construct_n<_ValueType>(_VSTD::move(__first), __n);
}

// uninitialized_value_construct

template <class _ValueType, class _ForwardIterator, class _Sentinel>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator __uninitialized_value_construct(_ForwardIterator __first, _Sentinel __last) {
    auto __idx = __first;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try {
#endif
    for (; __idx != __last; ++__idx)
        ::new (_VSTD::__voidify(*__idx)) _ValueType();
#ifndef _LIBCPP_NO_EXCEPTIONS
    } catch (...) {
        _VSTD::__destroy(__first, __idx);
        throw;
    }
#endif

    return __idx;
}

template <class _ForwardIterator>
inline _LIBCPP_HIDE_FROM_ABI
void uninitialized_value_construct(_ForwardIterator __first, _ForwardIterator __last) {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    (void)_VSTD::__uninitialized_value_construct<_ValueType>(
        _VSTD::move(__first), _VSTD::move(__last));
}

// uninitialized_value_construct_n

template <class _ValueType, class _ForwardIterator, class _Size>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator __uninitialized_value_construct_n(_ForwardIterator __first, _Size __n) {
    auto __idx = __first;
#ifndef _LIBCPP_NO_EXCEPTIONS
    try {
#endif
    for (; __n > 0; ++__idx, (void) --__n)
        ::new (_VSTD::__voidify(*__idx)) _ValueType();
#ifndef _LIBCPP_NO_EXCEPTIONS
    } catch (...) {
        _VSTD::__destroy(__first, __idx);
        throw;
    }
#endif

    return __idx;
}

template <class _ForwardIterator, class _Size>
inline _LIBCPP_HIDE_FROM_ABI
_ForwardIterator uninitialized_value_construct_n(_ForwardIterator __first, _Size __n) {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    return __uninitialized_value_construct_n<_ValueType>(_VSTD::move(__first), __n);
}

// uninitialized_move

template <class _ValueType, class _InputIterator, class _Sentinel1, class _ForwardIterator, class _Sentinel2,
          class _IterMove>
inline _LIBCPP_HIDE_FROM_ABI pair<_InputIterator, _ForwardIterator>
__uninitialized_move(_InputIterator __ifirst, _Sentinel1 __ilast,
                     _ForwardIterator __ofirst, _Sentinel2 __olast, _IterMove __iter_move) {
  auto __idx = __ofirst;
#ifndef _LIBCPP_NO_EXCEPTIONS
  try {
#endif
    for (; __ifirst != __ilast && __idx != __olast; ++__idx, (void)++__ifirst) {
      ::new (_VSTD::__voidify(*__idx)) _ValueType(__iter_move(__ifirst));
    }
#ifndef _LIBCPP_NO_EXCEPTIONS
  } catch (...) {
    _VSTD::__destroy(__ofirst, __idx);
    throw;
  }
#endif

  return {_VSTD::move(__ifirst), _VSTD::move(__idx)};
}

template <class _InputIterator, class _ForwardIterator>
inline _LIBCPP_HIDE_FROM_ABI _ForwardIterator uninitialized_move(_InputIterator __ifirst, _InputIterator __ilast,
                                                                 _ForwardIterator __ofirst) {
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  auto __iter_move = [](auto&& __iter) -> decltype(auto) { return _VSTD::move(*__iter); };

  auto __result = _VSTD::__uninitialized_move<_ValueType>(_VSTD::move(__ifirst), _VSTD::move(__ilast),
                                                          _VSTD::move(__ofirst), __unreachable_sentinel(), __iter_move);
  return _VSTD::move(__result.second);
}

// uninitialized_move_n

template <class _ValueType, class _InputIterator, class _Size, class _ForwardIterator, class _Sentinel, class _IterMove>
inline _LIBCPP_HIDE_FROM_ABI pair<_InputIterator, _ForwardIterator>
__uninitialized_move_n(_InputIterator __ifirst, _Size __n,
                       _ForwardIterator __ofirst, _Sentinel __olast, _IterMove __iter_move) {
  auto __idx = __ofirst;
#ifndef _LIBCPP_NO_EXCEPTIONS
  try {
#endif
    for (; __n > 0 && __idx != __olast; ++__idx, (void)++__ifirst, --__n)
      ::new (_VSTD::__voidify(*__idx)) _ValueType(__iter_move(__ifirst));
#ifndef _LIBCPP_NO_EXCEPTIONS
  } catch (...) {
    _VSTD::__destroy(__ofirst, __idx);
    throw;
  }
#endif

  return {_VSTD::move(__ifirst), _VSTD::move(__idx)};
}

template <class _InputIterator, class _Size, class _ForwardIterator>
inline _LIBCPP_HIDE_FROM_ABI pair<_InputIterator, _ForwardIterator>
uninitialized_move_n(_InputIterator __ifirst, _Size __n, _ForwardIterator __ofirst) {
  using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
  auto __iter_move = [](auto&& __iter) -> decltype(auto) { return _VSTD::move(*__iter); };

  return _VSTD::__uninitialized_move_n<_ValueType>(_VSTD::move(__ifirst), __n, _VSTD::move(__ofirst),
                                                   __unreachable_sentinel(), __iter_move);
}

// Destroys every element in the range [first, last) FROM RIGHT TO LEFT using allocator
// destruction. If elements are themselves C-style arrays, they are recursively destroyed
// in the same manner.
//
// This function assumes that destructors do not throw, and that the allocator is bound to
// the correct type.
template<class _Alloc, class _BidirIter, class = __enable_if_t<
    __is_cpp17_bidirectional_iterator<_BidirIter>::value
>>
_LIBCPP_HIDE_FROM_ABI
constexpr void __allocator_destroy_multidimensional(_Alloc& __alloc, _BidirIter __first, _BidirIter __last) noexcept {
    using _ValueType = typename iterator_traits<_BidirIter>::value_type;
    static_assert(is_same_v<typename allocator_traits<_Alloc>::value_type, _ValueType>,
        "The allocator should already be rebound to the correct type");

    if (__first == __last)
        return;

    if constexpr (is_array_v<_ValueType>) {
        static_assert(!__libcpp_is_unbounded_array<_ValueType>::value,
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
template<class _Alloc, class _Tp>
_LIBCPP_HIDE_FROM_ABI
constexpr void __allocator_construct_at(_Alloc& __alloc, _Tp* __loc) {
    static_assert(is_same_v<typename allocator_traits<_Alloc>::value_type, _Tp>,
        "The allocator should already be rebound to the correct type");

    if constexpr (is_array_v<_Tp>) {
        using _Element = remove_extent_t<_Tp>;
        __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
        size_t __i = 0;
        _Tp& __array = *__loc;

        // If an exception is thrown, destroy what we have constructed so far in reverse order.
        __transaction __guard([&]() { std::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + __i); });
        for (; __i != extent_v<_Tp>; ++__i) {
            std::__allocator_construct_at(__elem_alloc, std::addressof(__array[__i]));
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
template<class _Alloc, class _Tp, class _Arg>
_LIBCPP_HIDE_FROM_ABI
constexpr void __allocator_construct_at(_Alloc& __alloc, _Tp* __loc, _Arg const& __arg) {
    static_assert(is_same_v<typename allocator_traits<_Alloc>::value_type, _Tp>,
        "The allocator should already be rebound to the correct type");

    if constexpr (is_array_v<_Tp>) {
        static_assert(is_array_v<_Arg>,
            "Provided non-array initialization argument to __allocator_construct_at when "
            "trying to construct an array.");

        using _Element = remove_extent_t<_Tp>;
        __allocator_traits_rebind_t<_Alloc, _Element> __elem_alloc(__alloc);
        size_t __i = 0;
        _Tp& __array = *__loc;

        // If an exception is thrown, destroy what we have constructed so far in reverse order.
        __transaction __guard([&]() { std::__allocator_destroy_multidimensional(__elem_alloc, __array, __array + __i); });
        for (; __i != extent_v<_Tp>; ++__i) {
            std::__allocator_construct_at(__elem_alloc, std::addressof(__array[__i]), __arg[__i]);
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
template<class _Alloc, class _BidirIter, class _Tp, class _Size = typename iterator_traits<_BidirIter>::difference_type>
_LIBCPP_HIDE_FROM_ABI
constexpr void __uninitialized_allocator_fill_n(_Alloc& __alloc, _BidirIter __it, _Size __n, _Tp const& __value) {
    using _ValueType = typename iterator_traits<_BidirIter>::value_type;
    __allocator_traits_rebind_t<_Alloc, _ValueType> __value_alloc(__alloc);
    _BidirIter __begin = __it;

    // If an exception is thrown, destroy what we have constructed so far in reverse order.
    __transaction __guard([&]() { std::__allocator_destroy_multidimensional(__value_alloc, __begin, __it); });
    for (; __n != 0; --__n, ++__it) {
        std::__allocator_construct_at(__value_alloc, std::addressof(*__it), __value);
    }
    __guard.__complete();
}

// Same as __uninitialized_allocator_fill_n, but doesn't pass any initialization argument
// to the allocator's construct method, which results in value initialization.
template<class _Alloc, class _BidirIter, class _Size = typename iterator_traits<_BidirIter>::difference_type>
_LIBCPP_HIDE_FROM_ABI
constexpr void __uninitialized_allocator_value_construct_n(_Alloc& __alloc, _BidirIter __it, _Size __n) {
    using _ValueType = typename iterator_traits<_BidirIter>::value_type;
    __allocator_traits_rebind_t<_Alloc, _ValueType> __value_alloc(__alloc);
    _BidirIter __begin = __it;

    // If an exception is thrown, destroy what we have constructed so far in reverse order.
    __transaction __guard([&]() { std::__allocator_destroy_multidimensional(__value_alloc, __begin, __it); });
    for (; __n != 0; --__n, ++__it) {
        std::__allocator_construct_at(__value_alloc, std::addressof(*__it));
    }
    __guard.__complete();
}

#endif // _LIBCPP_STD_VER > 14

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_UNINITIALIZED_ALGORITHMS_H
