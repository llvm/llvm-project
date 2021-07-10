//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <optional>

// class bad_optional_access : public exception

#include <optional>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    using std::bad_optional_access;

    static_assert(std::is_base_of<std::exception, bad_optional_access>::value, "");
    static_assert(std::is_convertible<bad_optional_access*, std::exception*>::value, "");

  return 0;
}
