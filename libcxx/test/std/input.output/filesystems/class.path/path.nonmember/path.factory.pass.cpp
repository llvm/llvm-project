//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// template <class Source>
//    path u8path(Source const&);
// template <class InputIter>
//   path u8path(InputIter, InputIter);

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.hpp"
#include "filesystem_test_helper.hpp"


int main(int, char**)
{
  using namespace fs;
  const char* In1 = "abcd/efg";
  const std::string In2(In1);
  const auto In3 = In2.begin();
  const auto In3End = In2.end();
  {
    path p = fs::u8path(In1);
    assert(p == In1);
  }
  {
    path p = fs::u8path(In2);
    assert(p == In1);
  }
  {
    path p = fs::u8path(In3);
    assert(p == In1);
  }
  {
    path p = fs::u8path(In3, In3End);
    assert(p == In1);
  }

  return 0;
}
