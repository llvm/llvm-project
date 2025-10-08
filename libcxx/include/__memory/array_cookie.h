// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ARRAY_COOKIE_H
#define _LIBCPP___MEMORY_ARRAY_COOKIE_H

#include <__config>
#include <__configuration/abi.h>
#include <__cstddef/size_t.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__type_traits/negation.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Trait representing whether a type requires an array cookie at the start of its allocation when
// allocated as `new T[n]` and deallocated as `delete[] array`.
//
// Under the Itanium C++ ABI [1] and the ARM ABI which derives from it, we know that an array cookie is available
// unless `T` is trivially destructible and the call to `operator delete[]` is not a sized operator delete. Under
// other ABIs, we assume there are no array cookies.
//
// [1]: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#array-cookies
#if defined(_LIBCPP_ABI_ITANIUM) || defined(_LIBCPP_ABI_ARM_WITH_32BIT_ODDITIES)
// TODO: Use a builtin instead
// TODO: We should factor in the choice of the usual deallocation function in this determination:
//       a cookie may be available in more cases but we ignore those for now.
template <class _Tp>
struct __has_array_cookie : _Not<is_trivially_destructible<_Tp> > {};
#else
template <class _Tp>
struct __has_array_cookie : false_type {};
#endif

// Return the array cookie located before the given pointer.
//
// In the Itanium ABI [1]
// ----------------------
// The array cookie is stored immediately before the first element of the array. If the preferred alignment
// of array elements (which is different from the ABI alignment) is more than that of size_t, additional
// padding bytes exist before the array cookie. Assuming array elements of size and alignment 16 bytes, that
// gives us the following layout:
//
// |ooooooooxxxxxxxxaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbccccccccccccccccdddddddddddddddd|
//  ^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//     |    ^^^^^^^^                               |
//     |       |                              array elements
//  padding    |
//        array cookie
//
// In practice, it is sufficient to read the bytes immediately before the first array element.
//
//
// In the ARM ABI [2]
// ------------------
// The array cookie is stored at the very start of the allocation and it has the following form:
//
//    struct array_cookie {
//      std::size_t element_size; // element_size != 0
//      std::size_t element_count;
//    };
//
// Assuming elements of size and alignment 32 bytes, this gives us the following layout:
//
//  |xxxxxxxxXXXXXXXXooooooooooooooooaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb|
//   ^^^^^^^^        ^^^^^^^^^^^^^^^^
//      |    ^^^^^^^^        |       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// element size  |        padding                                 |
//         element count                                     array elements
//
// We calculate the starting address of the allocation by taking into account the ABI (not the preferred)
// alignment of the type.
//
// [1]: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#array-cookies
// [2]: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Handle-C++-differences
template <class _Tp>
// Avoid failures when -fsanitize-address-poison-custom-array-cookie is enabled
_LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_SANITIZE("address") size_t __get_array_cookie(_Tp const* __ptr) {
  static_assert(
      __has_array_cookie<_Tp>::value, "Trying to access the array cookie of a type that is not guaranteed to have one");

#if defined(_LIBCPP_ABI_ITANIUM)

  size_t const* __cookie = reinterpret_cast<size_t const*>(__ptr) - 1;
  return *__cookie;

#elif defined(_LIBCPP_ABI_ARM_WITH_32BIT_ODDITIES)

  struct _ArrayCookie {
    size_t __element_size;
    size_t __element_count;
  };

  size_t __cookie_size_with_padding = // max(sizeof(_ArrayCookie), alignof(T))
      sizeof(_ArrayCookie) < alignof(_Tp) ? alignof(_Tp) : sizeof(_ArrayCookie);
  char const* __allocation_start = reinterpret_cast<char const*>(__ptr) - __cookie_size_with_padding;
  _ArrayCookie const* __cookie   = reinterpret_cast<_ArrayCookie const*>(__allocation_start);
  return __cookie->__element_count;

#else

  static_assert(sizeof(_Tp) == 0, "This function is not implemented for this ABI");

#endif
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ARRAY_COOKIE_H
