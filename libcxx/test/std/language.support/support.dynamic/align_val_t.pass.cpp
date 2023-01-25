//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// enum class align_val_t : size_t {}

// UNSUPPORTED: c++03, c++11, c++14

// Libcxx when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

#include <new>
#include <type_traits>

#include "test_macros.h"

int main(int, char**) {
  {
    static_assert(std::is_enum<std::align_val_t>::value, "");
    static_assert(std::is_same<std::underlying_type<std::align_val_t>::type, std::size_t>::value, "");
    static_assert(!std::is_constructible<std::align_val_t, std::size_t>::value, "");
    static_assert(!std::is_constructible<std::size_t, std::align_val_t>::value, "");
  }
  {
    constexpr auto a = std::align_val_t(0);
    constexpr auto b = std::align_val_t(32);
    constexpr auto c = std::align_val_t(-1);
    static_assert(a != b, "");
    static_assert(a == std::align_val_t(0), "");
    static_assert(b == std::align_val_t(32), "");
    static_assert(static_cast<std::size_t>(c) == (std::size_t)-1, "");
  }

  return 0;
}
