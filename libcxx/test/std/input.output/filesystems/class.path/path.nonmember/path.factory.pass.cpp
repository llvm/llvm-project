//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <filesystem>

// template <class Source>
//    path u8path(Source const&);
// template <class InputIter>
//   path u8path(InputIter, InputIter);

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


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
    path p = fs::u8path(In2.data());
    assert(p == In1);
  }
  {
    path p = fs::u8path(In3, In3End);
    assert(p == In1);
  }
#if TEST_STD_VER > 17 && defined(__cpp_char8_t) && defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_LOCALIZATION)
  const char8_t* u8In1 = u8"abcd/efg";
  const std::u8string u8In2(u8In1);
  const auto u8In3 = u8In2.begin();
  const auto u8In3End = u8In2.end();
  // Proposed in P1423, marked tested only for libc++
  {
    path p = fs::u8path(u8In1);
    assert(p == In1);
  }
  {
    path p = fs::u8path(u8In2);
    assert(p == In1);
  }
  {
    path p = fs::u8path(u8In2.data());
    assert(p == In1);
  }
  {
    path p = fs::u8path(u8In3, u8In3End);
    assert(p == In1);
  }
#endif

  return 0;
}
