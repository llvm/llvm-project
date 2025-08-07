//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// enum class align_val_t : size_t {}

// UNSUPPORTED: c++03, c++11, c++14

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// XFAIL: target={{.+}}-zos{{.*}}

#include <new>
#include <cassert>
#include <cstddef>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "test_macros.h"

constexpr bool test() {
  static_assert(std::is_enum<std::align_val_t>::value, "");
  static_assert(std::is_same<std::underlying_type<std::align_val_t>::type, std::size_t>::value, "");
  static_assert(!std::is_constructible<std::align_val_t, std::size_t>::value, "");
  static_assert(!std::is_constructible<std::size_t, std::align_val_t>::value, "");

  {
    auto a = std::align_val_t(0);
    auto b = std::align_val_t(32);
    auto c = std::align_val_t(-1);
    assert(a != b);
    assert(a == std::align_val_t(0));
    assert(b == std::align_val_t(32));
    assert(static_cast<std::size_t>(c) == static_cast<std::size_t>(-1));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_RTTI)
  {
    // Check that libc++ doesn't define align_val_t in a versioning namespace.
    // And that it mangles the same in C++03 through C++17
#  ifdef _MSC_VER
    // MSVC uses a different C++ ABI with a different name mangling scheme.
    // The type id name doesn't seem to contain the mangled form at all.
    assert(typeid(std::align_val_t).name() == std::string("enum std::align_val_t"));
#  else
    assert(typeid(std::align_val_t).name() == std::string("St11align_val_t"));
#  endif
  }
#endif

  return 0;
}
