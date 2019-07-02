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
//   constexpr T ceil2(T x) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <bit>
#include <cstdint>
#include <limits>
#include <cassert>

#include "test_macros.h"

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <typename T>
constexpr bool toobig()
{
	return 0 == std::ceil2(std::numeric_limits<T>::max());
}

int main()
{
//	Make sure we generate a compile-time error for UB
	static_assert(toobig<unsigned char>(),      ""); // expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<unsigned short>(),     ""); // expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<unsigned>(),           ""); // expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<unsigned long>(),      ""); // expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<unsigned long long>(), ""); // expected-error {{static_assert expression is not an integral constant expression}}

	static_assert(toobig<uint8_t>(), ""); 	// expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<uint16_t>(), ""); 	// expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<uint32_t>(), ""); 	// expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<uint64_t>(), ""); 	// expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<size_t>(), ""); 	// expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<uintmax_t>(), "");	// expected-error {{static_assert expression is not an integral constant expression}}
	static_assert(toobig<uintptr_t>(), "");	// expected-error {{static_assert expression is not an integral constant expression}}
}
