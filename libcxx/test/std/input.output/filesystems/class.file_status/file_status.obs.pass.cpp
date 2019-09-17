//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// class file_status

// file_type type() const noexcept;
// perms permissions(p) const noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"


int main(int, char**) {
  using namespace fs;

  const file_status st(file_type::regular, perms::owner_read);

  // type test
  {
    static_assert(noexcept(st.type()),
                  "operation must be noexcept");
    static_assert(std::is_same<decltype(st.type()), file_type>::value,
                 "operation must return file_type");
    assert(st.type() == file_type::regular);
  }
  // permissions test
  {
    static_assert(noexcept(st.permissions()),
                  "operation must be noexcept");
    static_assert(std::is_same<decltype(st.permissions()), perms>::value,
                 "operation must return perms");
    assert(st.permissions() == perms::owner_read);
  }

  return 0;
}
