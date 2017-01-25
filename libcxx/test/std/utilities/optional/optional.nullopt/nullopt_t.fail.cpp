//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// struct nullopt_t{see below};
// constexpr nullopt_t nullopt(unspecified);

// [optional.nullopt]/2:
//   Type nullopt_t shall not have a default constructor or an initializer-list constructor.
//   It shall not be an aggregate and shall be a literal type. 
//   Constant nullopt shall be initialized with an argument of literal type.

#include <optional>
#include "test_macros.h"

int main()
{
	std::nullopt_t n = {};
}
