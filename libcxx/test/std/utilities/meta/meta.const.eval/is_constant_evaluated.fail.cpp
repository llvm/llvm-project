//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <type_traits>


#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifndef __cpp_lib_is_constant_evaluated
  // expected-error@+1 {{no member named 'is_constant_evaluated' in namespace 'std'}}
  bool b = std::is_constant_evaluated();
#else
  // expected-error@+1 {{static_assert failed}}
  static_assert(!std::is_constant_evaluated(), "");
#endif
  return 0;
}
