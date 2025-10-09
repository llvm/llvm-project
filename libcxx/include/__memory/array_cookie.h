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
#if defined(_LIBCPP_ABI_ITANIUM) || defined(_LIBCPP_ABI_ITANIUM_WITH_ARM_DIFFERENCES)
// TODO: Use a builtin instead
// TODO: We should factor in the choice of the usual deallocation function in this determination:
//       a cookie may be available in more cases but we ignore those for now.
template <class _Tp>
struct __has_array_cookie : _Not<is_trivially_destructible<_Tp> > {};
#else
template <class _Tp>
struct __has_array_cookie : false_type {};
#endif

template <class _Tp, bool _HasPadding = (_LIBCPP_PREFERRED_ALIGNOF(_Tp) > sizeof(size_t))>
struct [[__gnu__::__aligned__(_LIBCPP_PREFERRED_ALIGNOF(_Tp))]] __itanium_array_cookie {
  size_t __element_count;
};

template <class _Tp>
struct [[__gnu__::__aligned__(_LIBCPP_PREFERRED_ALIGNOF(_Tp))]] __itanium_array_cookie<_Tp, /* _HasPadding */ true> {
  char __padding[_LIBCPP_PREFERRED_ALIGNOF(_Tp) - sizeof(size_t)];
  size_t __element_count;
};

template <class _Tp>
struct [[__gnu__::__aligned__(_LIBCPP_ALIGNOF(_Tp))]] __arm_array_cookie {
  size_t __element_size;
  size_t __element_count;
};

// Return the element count in the array cookie located before the given pointer.
//
// In the Itanium ABI [1]
// ----------------------
// The element count is stored immediately before the first element of the array. If the preferred alignment
// of array elements (which is different from the ABI alignment) is more than that of size_t, additional
// padding bytes exist at the beginning of the array cookie. Assuming array elements of size and alignment 16
// bytes, that gives us the following layout:
//
// |ooooooooxxxxxxxxaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbccccccccccccccccdddddddddddddddd|
//  ^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//     |    ^^^^^^^^                               |
//     |       |                              array elements
//  padding    |
//       element count
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
// We calculate the starting address of the allocation by taking into account the ABI alignment (not
// the preferred alignment) of the type.
//
// [1]: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#array-cookies
// [2]: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Handle-C++-differences
template <class _Tp>
// Avoid failures when -fsanitize-address-poison-custom-array-cookie is enabled
_LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_SANITIZE("address") size_t __get_array_cookie([[__maybe_unused__]] _Tp const* __ptr) {
  static_assert(
      __has_array_cookie<_Tp>::value, "Trying to access the array cookie of a type that is not guaranteed to have one");

#if defined(_LIBCPP_ABI_ITANIUM)
  using _ArrayCookie = __itanium_array_cookie<_Tp>;
#elif defined(_LIBCPP_ABI_ITANIUM_WITH_ARM_DIFFERENCES)
  using _ArrayCookie = __arm_array_cookie<_Tp>;
#else
  static_assert(false, "The array cookie layout is unknown on this ABI");
#endif

  char const* __allocation_start = reinterpret_cast<char const*>(__ptr) - sizeof(_ArrayCookie);
  _ArrayCookie __cookie;
  // This is necessary to avoid violating strict aliasing. It's valid because _ArrayCookie is an
  // implicit lifetime type.
  __builtin_memcpy(&__cookie, __allocation_start, sizeof(_ArrayCookie));
  return __cookie.__element_count;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ARRAY_COOKIE_H
