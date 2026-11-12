//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// class file_status

// void type(file_type) noexcept;
// void permissions(perms) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
namespace fs = std::filesystem;

int main(int, char**) {
  using namespace fs;

  file_status st;

  // type test
  {
    static_assert(noexcept(st.type(file_type::regular)),
                  "operation must be noexcept");
    static_assert(std::is_same<decltype(st.type(file_type::regular)), void>::value,
                 "operation must return void");
    assert(st.type() != file_type::regular);
    st.type(file_type::regular);
    assert(st.type() == file_type::regular);
  }
  // permissions test
  {
    static_assert(noexcept(st.permissions(perms::owner_read)),
                  "operation must be noexcept");
    static_assert(std::is_same<decltype(st.permissions(perms::owner_read)), void>::value,
                 "operation must return void");
    assert(st.permissions() != perms::owner_read);
    st.permissions(perms::owner_read);
    assert(st.permissions() == perms::owner_read);
  }

  return 0;
}
