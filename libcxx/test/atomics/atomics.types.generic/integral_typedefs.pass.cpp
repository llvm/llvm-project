//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// typedef atomic<char>               atomic_char;
// typedef atomic<signed char>        atomic_schar;
// typedef atomic<unsigned char>      atomic_uchar;
// typedef atomic<short>              atomic_short;
// typedef atomic<unsigned short>     atomic_ushort;
// typedef atomic<int>                atomic_int;
// typedef atomic<unsigned int>       atomic_uint;
// typedef atomic<long>               atomic_long;
// typedef atomic<unsigned long>      atomic_ulong;
// typedef atomic<long long>          atomic_llong;
// typedef atomic<unsigned long long> atomic_ullong;
// typedef atomic<char16_t>           atomic_char16_t;
// typedef atomic<char32_t>           atomic_char32_t;
// typedef atomic<wchar_t>            atomic_wchar_t;

#include <atomic>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::atomic<char>, std::atomic_char>::value), "");
    static_assert((std::is_same<std::atomic<signed char>, std::atomic_schar>::value), "");
    static_assert((std::is_same<std::atomic<unsigned char>, std::atomic_uchar>::value), "");
    static_assert((std::is_same<std::atomic<short>, std::atomic_short>::value), "");
    static_assert((std::is_same<std::atomic<unsigned short>, std::atomic_ushort>::value), "");
    static_assert((std::is_same<std::atomic<int>, std::atomic_int>::value), "");
    static_assert((std::is_same<std::atomic<unsigned int>, std::atomic_uint>::value), "");
    static_assert((std::is_same<std::atomic<long>, std::atomic_long>::value), "");
    static_assert((std::is_same<std::atomic<unsigned long>, std::atomic_ulong>::value), "");
    static_assert((std::is_same<std::atomic<long long>, std::atomic_llong>::value), "");
    static_assert((std::is_same<std::atomic<unsigned long long>, std::atomic_ullong>::value), "");
    static_assert((std::is_same<std::atomic<wchar_t>, std::atomic_wchar_t>::value), "");
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    static_assert((std::is_same<std::atomic<char16_t>, std::atomic_char16_t>::value), "");
    static_assert((std::is_same<std::atomic<char32_t>, std::atomic_char32_t>::value), "");
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
