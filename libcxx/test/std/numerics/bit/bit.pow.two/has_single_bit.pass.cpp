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
//   constexpr bool has_single_bit(T x) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

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
	return  std::has_single_bit(T(1))
	   &&   std::has_single_bit(T(2))
	   &&  !std::has_single_bit(T(3))
	   &&   std::has_single_bit(T(4))
	   &&  !std::has_single_bit(T(5))
	   &&  !std::has_single_bit(T(6))
	   &&  !std::has_single_bit(T(7))
	   &&   std::has_single_bit(T(8))
	   &&  !std::has_single_bit(T(9))
	   ;
}


template <typename T>
void runtime_test()
{
	ASSERT_SAME_TYPE(bool, decltype(std::has_single_bit(T(0))));
	ASSERT_NOEXCEPT(                std::has_single_bit(T(0)));

	assert(!std::has_single_bit(T(121)));
	assert(!std::has_single_bit(T(122)));
	assert(!std::has_single_bit(T(123)));
	assert(!std::has_single_bit(T(124)));
	assert(!std::has_single_bit(T(125)));
	assert(!std::has_single_bit(T(126)));
	assert(!std::has_single_bit(T(127)));
	assert( std::has_single_bit(T(128)));
	assert(!std::has_single_bit(T(129)));
	assert(!std::has_single_bit(T(130)));
}

int main(int, char**)
{

    {
    auto lambda = [](auto x) -> decltype(std::has_single_bit(x)) {};
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
    static_assert(!std::is_invocable_v<L, __int128_t>, "");
#endif

    static_assert(!std::is_invocable_v<L, A>, "");
    static_assert(!std::is_invocable_v<L, E1>, "");
    static_assert(!std::is_invocable_v<L, E2>, "");
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
	__uint128_t val = 128;
	val <<= 32;
	assert(!std::has_single_bit(val-1));
	assert( std::has_single_bit(val));
	assert(!std::has_single_bit(val+1));
	val <<= 2;
	assert(!std::has_single_bit(val-1));
	assert( std::has_single_bit(val));
	assert(!std::has_single_bit(val+1));
	val <<= 3;
	assert(!std::has_single_bit(val-1));
	assert( std::has_single_bit(val));
	assert(!std::has_single_bit(val+1));
	}
#endif

    return 0;
}
