//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS

// type_traits

// is_literal_type

#include <type_traits>

void f() {
  [[maybe_unused]] std::is_literal_type<int> a; // expected-warning {{'is_literal_type<int>' is deprecated}}
}
