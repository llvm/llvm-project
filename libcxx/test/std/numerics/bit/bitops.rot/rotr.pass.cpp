//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int rotr(T x, unsigned int s) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//  T is an unsigned integer type

#include <bit>
#include <cstdint>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <typename T>
constexpr bool constexpr_test()
{
    const T max = std::numeric_limits<T>::max();

    return std::rotr(T(128), 0) == T(128)
       &&  std::rotr(T(128), 1) == T( 64)
       &&  std::rotr(T(128), 2) == T( 32)
       &&  std::rotr(T(128), 3) == T( 16)
       &&  std::rotr(T(128), 4) == T(  8)
       &&  std::rotr(T(128), 5) == T(  4)
       &&  std::rotr(T(128), 6) == T(  2)
       &&  std::rotr(T(128), 7) == T(  1)
       &&  std::rotr(max, 0)  == max
       &&  std::rotr(max, 1)  == max
       &&  std::rotr(max, 2)  == max
       &&  std::rotr(max, 3)  == max
       &&  std::rotr(max, 4)  == max
       &&  std::rotr(max, 5)  == max
       &&  std::rotr(max, 6)  == max
       &&  std::rotr(max, 7)  == max
      ;
}


template <typename T>
void runtime_test()
{
    ASSERT_SAME_TYPE(T, decltype(std::rotr(T(0), 0)));
    ASSERT_NOEXCEPT(             std::rotr(T(0), 0));
    const T max = std::numeric_limits<T>::max();
    const T val = std::numeric_limits<T>::max() - 1;

    const T uppers [] = {
        max,              // not used
        max - max,        // 000 .. 0
        max - (max >> 1), // 800 .. 0
        max - (max >> 2), // C00 .. 0
        max - (max >> 3), // E00 .. 0
        max - (max >> 4), // F00 .. 0
        max - (max >> 5), // F80 .. 0
        max - (max >> 6), // FC0 .. 0
        max - (max >> 7), // FE0 .. 0
        };

    assert( std::rotr(val, 0) == val);
    assert( std::rotr(val, 1) == T((val >> 1) +  uppers[1]));
    assert( std::rotr(val, 2) == T((val >> 2) +  uppers[2]));
    assert( std::rotr(val, 3) == T((val >> 3) +  uppers[3]));
    assert( std::rotr(val, 4) == T((val >> 4) +  uppers[4]));
    assert( std::rotr(val, 5) == T((val >> 5) +  uppers[5]));
    assert( std::rotr(val, 6) == T((val >> 6) +  uppers[6]));
    assert( std::rotr(val, 7) == T((val >> 7) +  uppers[7]));
}

int main()
{

    {
    auto lambda = [](auto x) -> decltype(std::rotr(x, 1U)) {};
    using L = decltype(lambda);

    static_assert( std::is_invocable_v<L, unsigned char>, "");
    static_assert( std::is_invocable_v<L, unsigned int>, "");
    static_assert( std::is_invocable_v<L, unsigned long>, "");
    static_assert( std::is_invocable_v<L, unsigned long long>, "");

    static_assert( std::is_invocable_v<L, uint8_t>, "");
    static_assert( std::is_invocable_v<L, uint16_t>, "");
    static_assert( std::is_invocable_v<L, uint32_t>, "");
    static_assert( std::is_invocable_v<L, uint64_t>, "");
    static_assert( std::is_invocable_v<L, size_t>, "");

    static_assert( std::is_invocable_v<L, uintmax_t>, "");
    static_assert( std::is_invocable_v<L, uintptr_t>, "");


    static_assert(!std::is_invocable_v<L, int>, "");
    static_assert(!std::is_invocable_v<L, signed int>, "");
    static_assert(!std::is_invocable_v<L, long>, "");
    static_assert(!std::is_invocable_v<L, long long>, "");

    static_assert(!std::is_invocable_v<L, int8_t>, "");
    static_assert(!std::is_invocable_v<L, int16_t>, "");
    static_assert(!std::is_invocable_v<L, int32_t>, "");
    static_assert(!std::is_invocable_v<L, int64_t>, "");
    static_assert(!std::is_invocable_v<L, ptrdiff_t>, "");

    static_assert(!std::is_invocable_v<L, bool>, "");
    static_assert(!std::is_invocable_v<L, signed char>, "");
    static_assert(!std::is_invocable_v<L, char16_t>, "");
    static_assert(!std::is_invocable_v<L, char32_t>, "");

#ifndef _LIBCPP_HAS_NO_INT128
    static_assert( std::is_invocable_v<L, __uint128_t>, "");
    static_assert(!std::is_invocable_v<L,  __int128_t>, "");
#endif

    static_assert(!std::is_invocable_v<L, A, unsigned>, "");
    static_assert(!std::is_invocable_v<L, E1, unsigned>, "");
    static_assert(!std::is_invocable_v<L, E2, unsigned>, "");
    }

    static_assert(constexpr_test<unsigned char>(),      "");
    static_assert(constexpr_test<unsigned short>(),     "");
    static_assert(constexpr_test<unsigned>(),           "");
    static_assert(constexpr_test<unsigned long>(),      "");
    static_assert(constexpr_test<unsigned long long>(), "");

    static_assert(constexpr_test<uint8_t>(),   "");
    static_assert(constexpr_test<uint16_t>(),  "");
    static_assert(constexpr_test<uint32_t>(),  "");
    static_assert(constexpr_test<uint64_t>(),  "");
    static_assert(constexpr_test<size_t>(),    "");
    static_assert(constexpr_test<uintmax_t>(), "");
    static_assert(constexpr_test<uintptr_t>(), "");

#ifndef _LIBCPP_HAS_NO_INT128
    static_assert(constexpr_test<__uint128_t>(),        "");
#endif


    runtime_test<unsigned char>();
    runtime_test<unsigned>();
    runtime_test<unsigned short>();
    runtime_test<unsigned long>();
    runtime_test<unsigned long long>();

    runtime_test<uint8_t>();
    runtime_test<uint16_t>();
    runtime_test<uint32_t>();
    runtime_test<uint64_t>();
    runtime_test<size_t>();
    runtime_test<uintmax_t>();
    runtime_test<uintptr_t>();


#ifndef _LIBCPP_HAS_NO_INT128
    runtime_test<__uint128_t>();

    {
    __uint128_t val = 168; // 0xA8 (aka 10101000)

    assert( std::rotr(val, 128) == 168);
    val <<= 32;
    assert( std::rotr(val,  32) == 168);
    val <<= 2;
    assert( std::rotr(val,  33) == 336);
    val <<= 3;
    assert( std::rotr(val,  38) ==  84);
    assert( std::rotr(val, 166) ==  84);
    }
#endif
}
