//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// template <class T>
//   constexpr int popcount(T x) noexcept;

// Returns: The number of bits set to one in the value of x.
//
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
	return std::popcount(T(0)) == 0
	   &&  std::popcount(T(1)) == 1
	   &&  std::popcount(T(2)) == 1
	   &&  std::popcount(T(3)) == 2
	   &&  std::popcount(T(4)) == 1
	   &&  std::popcount(T(5)) == 2
	   &&  std::popcount(T(6)) == 2
	   &&  std::popcount(T(7)) == 3
	   &&  std::popcount(T(8)) == 1
	   &&  std::popcount(T(9)) == 2
	   &&  std::popcount(std::numeric_limits<T>::max()) == std::numeric_limits<T>::digits
	  ;
}


template <typename T>
void runtime_test()
{
	ASSERT_SAME_TYPE(int, decltype(std::popcount(T(0))));
	ASSERT_NOEXCEPT(               std::popcount(T(0)));

	assert( std::popcount(T(121)) == 5);
	assert( std::popcount(T(122)) == 5);
	assert( std::popcount(T(123)) == 6);
	assert( std::popcount(T(124)) == 5);
	assert( std::popcount(T(125)) == 6);
	assert( std::popcount(T(126)) == 6);
	assert( std::popcount(T(127)) == 7);
	assert( std::popcount(T(128)) == 1);
	assert( std::popcount(T(129)) == 2);
	assert( std::popcount(T(130)) == 2);
}

int main()
{

    {
    auto lambda = [](auto x) -> decltype(std::popcount(x)) {};
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
	assert( std::popcount(val-1) == 39);
	assert( std::popcount(val)   ==  1);
	assert( std::popcount(val+1) ==  2);
	val <<= 2;
	assert( std::popcount(val-1) == 41);
	assert( std::popcount(val)   ==  1);
	assert( std::popcount(val+1) ==  2);
	val <<= 3;
	assert( std::popcount(val-1) == 44);
	assert( std::popcount(val)   ==  1);
	assert( std::popcount(val+1) ==  2);
	}
#endif
}
