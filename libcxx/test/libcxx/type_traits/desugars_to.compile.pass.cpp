//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: FROZEN-CXX03-HEADERS-FIXME

// This test requires variable templates
// UNSUPPORTED: gcc && c++11

#include <__type_traits/desugars_to.h>

struct Tag {};
struct Operation {};

namespace std {
template <>
bool const __desugars_to_v<Tag, Operation> = true;
}

void tests() {
  // Make sure that __desugars_to is false by default
  {
    struct Foo {};
    static_assert(!std::__desugars_to_v<Tag, Foo>, "");
  }

  // Make sure that __desugars_to bypasses const and ref qualifiers on the operation
  {
    static_assert(std::__desugars_to_v<Tag, Operation>, ""); // no quals
    static_assert(std::__desugars_to_v<Tag, Operation const>, "");

    static_assert(std::__desugars_to_v<Tag, Operation&>, "");
    static_assert(std::__desugars_to_v<Tag, Operation const&>, "");

    static_assert(std::__desugars_to_v<Tag, Operation&&>, "");
    static_assert(std::__desugars_to_v<Tag, Operation const&&>, "");
  }
}
