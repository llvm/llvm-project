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

// UNSUPPORTED: c++98, c++03, c++11, c++14

// MODULES_DEFINES: _LIBCPP_DISABLE_NODISCARD_AFTER_CXX17
#define _LIBCPP_DISABLE_NODISCARD_AFTER_CXX17
#include <__config>

_LIBCPP_NODISCARD_AFTER_CXX17 int foo() { return 6; }

int main ()
{
	foo();	// no error here!
}
