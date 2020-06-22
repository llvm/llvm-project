//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// template<CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);
// template<Returnable R, CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);

// https://bugs.llvm.org/show_bug.cgi?id=23141
#include <functional>
#include <type_traits>

#include "test_macros.h"

struct Fun
{
  template<typename T, typename U>
  void operator()(T &&, U &&) const
  {
    static_assert(std::is_same<U, int &>::value, "");
  }
};

int main(int, char**)
{
    std::bind(Fun{}, std::placeholders::_1, 42)("hello");

  return 0;
}
