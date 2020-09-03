// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr span() noexcept;
//
//  Remarks: This constructor shall not participate in overload resolution
//          unless Extent == 0 || Extent == dynamic_extent is true.


#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
  std::span<int, 2> s; // expected-error {{no matching constructor for initialization of 'std::span<int, 2>'}}

  return 0;
}
