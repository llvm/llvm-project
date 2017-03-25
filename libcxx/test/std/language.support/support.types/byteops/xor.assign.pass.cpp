//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <test_macros.h>

// UNSUPPORTED: c++98, c++03, c++11, c++14
// The following compilers don't like "std::byte b1{1}"
// XFAIL: clang-3.5, clang-3.6, clang-3.7, clang-3.8
// XFAIL: apple-clang-6, apple-clang-7, apple-clang-8.0

// constexpr byte& operator ^=(byte l, byte r) noexcept;


constexpr std::byte test(std::byte b1, std::byte b2) {
	std::byte bret = b1;
	return bret ^= b2;
	}


int main () {
	std::byte b;  // not constexpr, just used in noexcept check
	constexpr std::byte b1{1};
	constexpr std::byte b8{8};
	constexpr std::byte b9{9};

	static_assert(noexcept(b ^= b), "" );

	static_assert(std::to_integer<int>(test(b1, b8)) == 9, "");
	static_assert(std::to_integer<int>(test(b1, b9)) == 8, "");
	static_assert(std::to_integer<int>(test(b8, b9)) == 1, "");

	static_assert(std::to_integer<int>(test(b8, b1)) == 9, "");
	static_assert(std::to_integer<int>(test(b9, b1)) == 8, "");
	static_assert(std::to_integer<int>(test(b9, b8)) == 1, "");
}
