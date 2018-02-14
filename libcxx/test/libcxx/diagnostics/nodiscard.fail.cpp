// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_NODISCARD_AFTER_CXX17 works
//	#define _LIBCPP_NODISCARD_AFTER_CXX17 [[nodiscard]]

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7, clang-3.8

#include <__config>

_LIBCPP_NODISCARD_AFTER_CXX17 int foo() { return 6; }

int main ()
{
	foo();	// expected-error {{ignoring return value of function declared with 'nodiscard' attribute}}
}
