//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ALIASES_H
#define _LIBCPP___ATOMIC_ALIASES_H

#include <__atomic/atomic.h>
#include <__atomic/atomic_lock_free.h>
#include <__atomic/contention_t.h>
#include <__atomic/is_always_lock_free.h>
#include <__config>
#include <__type_traits/conditional.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

typedef atomic<bool>               atomic_bool;
typedef atomic<char>               atomic_char;
typedef atomic<signed char>        atomic_schar;
typedef atomic<unsigned char>      atomic_uchar;
typedef atomic<short>              atomic_short;
typedef atomic<unsigned short>     atomic_ushort;
typedef atomic<int>                atomic_int;
typedef atomic<unsigned int>       atomic_uint;
typedef atomic<long>               atomic_long;
typedef atomic<unsigned long>      atomic_ulong;
typedef atomic<long long>          atomic_llong;
typedef atomic<unsigned long long> atomic_ullong;
#ifndef _LIBCPP_HAS_NO_CHAR8_T
typedef atomic<char8_t>            atomic_char8_t;
#endif
typedef atomic<char16_t>           atomic_char16_t;
typedef atomic<char32_t>           atomic_char32_t;
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
typedef atomic<wchar_t>            atomic_wchar_t;
#endif

typedef atomic<int_least8_t>   atomic_int_least8_t;
typedef atomic<uint_least8_t>  atomic_uint_least8_t;
typedef atomic<int_least16_t>  atomic_int_least16_t;
typedef atomic<uint_least16_t> atomic_uint_least16_t;
typedef atomic<int_least32_t>  atomic_int_least32_t;
typedef atomic<uint_least32_t> atomic_uint_least32_t;
typedef atomic<int_least64_t>  atomic_int_least64_t;
typedef atomic<uint_least64_t> atomic_uint_least64_t;

typedef atomic<int_fast8_t>   atomic_int_fast8_t;
typedef atomic<uint_fast8_t>  atomic_uint_fast8_t;
typedef atomic<int_fast16_t>  atomic_int_fast16_t;
typedef atomic<uint_fast16_t> atomic_uint_fast16_t;
typedef atomic<int_fast32_t>  atomic_int_fast32_t;
typedef atomic<uint_fast32_t> atomic_uint_fast32_t;
typedef atomic<int_fast64_t>  atomic_int_fast64_t;
typedef atomic<uint_fast64_t> atomic_uint_fast64_t;

typedef atomic< int8_t>  atomic_int8_t;
typedef atomic<uint8_t>  atomic_uint8_t;
typedef atomic< int16_t> atomic_int16_t;
typedef atomic<uint16_t> atomic_uint16_t;
typedef atomic< int32_t> atomic_int32_t;
typedef atomic<uint32_t> atomic_uint32_t;
typedef atomic< int64_t> atomic_int64_t;
typedef atomic<uint64_t> atomic_uint64_t;

typedef atomic<intptr_t>  atomic_intptr_t;
typedef atomic<uintptr_t> atomic_uintptr_t;
typedef atomic<size_t>    atomic_size_t;
typedef atomic<ptrdiff_t> atomic_ptrdiff_t;
typedef atomic<intmax_t>  atomic_intmax_t;
typedef atomic<uintmax_t> atomic_uintmax_t;

// atomic_*_lock_free : prefer the contention type most highly, then the largest lock-free type

#ifdef __cpp_lib_atomic_is_always_lock_free
# define _LIBCPP_CONTENTION_LOCK_FREE ::std::__libcpp_is_always_lock_free<__cxx_contention_t>::__value
#else
# define _LIBCPP_CONTENTION_LOCK_FREE false
#endif

#if ATOMIC_LLONG_LOCK_FREE == 2
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, long long>          __libcpp_signed_lock_free;
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, unsigned long long> __libcpp_unsigned_lock_free;
#elif ATOMIC_INT_LOCK_FREE == 2
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, int>                __libcpp_signed_lock_free;
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, unsigned int>       __libcpp_unsigned_lock_free;
#elif ATOMIC_SHORT_LOCK_FREE == 2
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, short>              __libcpp_signed_lock_free;
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, unsigned short>     __libcpp_unsigned_lock_free;
#elif ATOMIC_CHAR_LOCK_FREE == 2
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, char>               __libcpp_signed_lock_free;
typedef __conditional_t<_LIBCPP_CONTENTION_LOCK_FREE, __cxx_contention_t, unsigned char>      __libcpp_unsigned_lock_free;
#else
    // No signed/unsigned lock-free types
#define _LIBCPP_NO_LOCK_FREE_TYPES
#endif

#if !defined(_LIBCPP_NO_LOCK_FREE_TYPES)
typedef atomic<__libcpp_signed_lock_free> atomic_signed_lock_free;
typedef atomic<__libcpp_unsigned_lock_free> atomic_unsigned_lock_free;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ALIASES_H
