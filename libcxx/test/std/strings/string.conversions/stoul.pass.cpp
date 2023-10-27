//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// unsigned long stoul(const string& str, size_t *idx = 0, int base = 10);
// unsigned long stoul(const wstring& str, size_t *idx = 0, int base = 10);

#include <string>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

int main(int, char**) {
  assert(std::stoul("0") == 0);
  assert(std::stoul("-0") == 0);
  assert(std::stoul(" 10") == 10);
  {
    std::size_t idx = 0;
    assert(std::stoul("10g", &idx, 16) == 16);
    assert(idx == 2);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    std::size_t idx = 0;
    try {
      (void)std::stoul("", &idx);
      assert(false);
    } catch (const std::invalid_argument&) {
      assert(idx == 0);
    }
  }
  {
    std::size_t idx = 0;
    try {
      (void)std::stoul("  - 8", &idx);
      assert(false);
    } catch (const std::invalid_argument&) {
      assert(idx == 0);
    }
  }
  {
    std::size_t idx = 0;
    try {
      (void)std::stoul("a1", &idx);
      assert(false);
    } catch (const std::invalid_argument&) {
      assert(idx == 0);
    }
  }
  {
    std::size_t idx = 0;
    try {
      // LWG#2009 and PR14919
      (void)std::stoul("9999999999999999999999999999999999999999999999999", &idx);
      assert(false);
    } catch (const std::out_of_range&) {
      assert(idx == 0);
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  assert(std::stoul(L"0") == 0);
  assert(std::stoul(L"-0") == 0);
  assert(std::stoul(L" 10") == 10);
  {
    std::size_t idx = 0;
    assert(std::stoul(L"10g", &idx, 16) == 16);
    assert(idx == 2);
  }
#  ifndef TEST_HAS_NO_EXCEPTIONS
  {
    std::size_t idx = 0;
    try {
      (void)std::stoul(L"", &idx);
      assert(false);
    } catch (const std::invalid_argument&) {
      assert(idx == 0);
    }
  }
  {
    std::size_t idx = 0;
    try {
      (void)std::stoul(L"  - 8", &idx);
      assert(false);
    } catch (const std::invalid_argument&) {
      assert(idx == 0);
    }
  }
  {
    std::size_t idx = 0;
    try {
      (void)std::stoul(L"a1", &idx);
      assert(false);
    } catch (const std::invalid_argument&) {
      assert(idx == 0);
    }
  }
  {
    std::size_t idx = 0;
    try {
      // LWG#2009 and PR14919
      (void)std::stoul(L"9999999999999999999999999999999999999999999999999", &idx);
      assert(false);
    } catch (const std::out_of_range&) {
      assert(idx == 0);
    }
  }
#  endif // TEST_HAS_NO_EXCEPTIONS
#endif   // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
