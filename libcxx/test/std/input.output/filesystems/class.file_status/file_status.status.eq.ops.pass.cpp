//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <filesystem>

// class file_status

// friend bool operator==(const file_status& lhs, const file_status& rhs) noexcept
//   { return lhs.type() == rhs.type() && lhs.permissions() == rhs.permissions(); } // C++20

#include <cassert>
#include <filesystem>

#include "test_comparisons.h"

void test() {
  {
    std::filesystem::file_status f1;
    std::filesystem::file_status f2;

    assert(testEquality(f1, f2, true));
  }
  {
    std::filesystem::file_status f1{std::filesystem::file_type::regular, std::filesystem::perms::owner_read};
    std::filesystem::file_status f2{std::filesystem::file_type::regular, std::filesystem::perms::owner_read};

    assert(testEquality(f1, f2, true));
  }
  {
    std::filesystem::file_status f1{std::filesystem::file_type::regular, std::filesystem::perms::owner_read};
    std::filesystem::file_status f2{std::filesystem::file_type::none, std::filesystem::perms::owner_read};

    assert(testEquality(f1, f2, false));
  }
  {
    std::filesystem::file_status f1{std::filesystem::file_type::regular, std::filesystem::perms::owner_read};
    std::filesystem::file_status f2{std::filesystem::file_type::regular, std::filesystem::perms::owner_write};

    assert(testEquality(f1, f2, false));
  }
}

int main(int, char**) {
  test();

  return 0;
}
