//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// Doesn't pass due to use of is_trivially_* trait.
// XFAIL: gcc-4.9

// <utility>

// template <class T1, class T2> struct pair

// ~pair()


#include <utility>
#include <type_traits>
#include <string>
#include <cassert>

#include "test_macros.h"

int main()
{
  static_assert((std::is_trivially_destructible<
      std::pair<int, float> >::value), "");
  static_assert((!std::is_trivially_destructible<
      std::pair<int, std::string> >::value), "");
}
