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
//   constexpr int countl_zero(T x) noexcept;

// Returns: The number of consecutive 0 bits, starting from the most significant bit.
//   [ Note: Returns N if x == 0. ]
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
	const int dig = std::numeric_limits<T>::digits;
	return std::countl_zero(T(0)) == dig
	   &&  std::countl_zero(T(1)) == dig - 1
	   &&  std::countl_zero(T(2)) == dig - 2
	   &&  std::countl_zero(T(3)) == dig - 2
	   &&  std::countl_zero(T(4)) == dig - 3
	   &&  std::countl_zero(T(5)) == dig - 3
	   &&  std::countl_zero(T(6)) == dig - 3
	   &&  std::countl_zero(T(7)) == dig - 3
	   &&  std::countl_zero(T(8)) == dig - 4
	   &&  std::countl_zero(T(9)) == dig - 4
	   &&  std::countl_zero(std::numeric_limits<T>::max()) == 0
	  ;
}


template <typename T>
void runtime_test()
{
	ASSERT_SAME_TYPE(int, decltype(std::countl_zero(T(0))));
	ASSERT_NOEXCEPT(               std::countl_zero(T(0)));
	const int dig = std::numeric_limits<T>::digits;

	assert( std::countl_zero(T(121)) == dig - 7);
	assert( std::countl_zero(T(122)) == dig - 7);
	assert( std::countl_zero(T(123)) == dig - 7);
	assert( std::countl_zero(T(124)) == dig - 7);
	assert( std::countl_zero(T(125)) == dig - 7);
	assert( std::countl_zero(T(126)) == dig - 7);
	assert( std::countl_zero(T(127)) == dig - 7);
	assert( std::countl_zero(T(128)) == dig - 8);
	assert( std::countl_zero(T(129)) == dig - 8);
	assert( std::countl_zero(T(130)) == dig - 8);
}

int main(int, char**)
{

    {
    auto lambda = [](auto x) -> decltype(std::countl_zero(x)) {};
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
	const int dig = std::numeric_limits<__uint128_t>::digits;
	__uint128_t val = 128;

	val <<= 32;
	assert( std::countl_zero(val-1) == dig - 39);
	assert( std::countl_zero(val)   == dig - 40);
	assert( std::countl_zero(val+1) == dig - 40);
	val <<= 2;
	assert( std::countl_zero(val-1) == dig - 41);
	assert( std::countl_zero(val)   == dig - 42);
	assert( std::countl_zero(val+1) == dig - 42);
	val <<= 3;
	assert( std::countl_zero(val-1) == dig - 44);
	assert( std::countl_zero(val)   == dig - 45);
	assert( std::countl_zero(val+1) == dig - 45);
	}
#endif

    return 0;
}
