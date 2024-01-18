//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr>
// struct pointer_traits
// {
//     typedef <details> pointer;
//     ...
// };

#include <memory>
#include <type_traits>

struct Foo {
  using element_type = int;
};

static_assert(std::is_same<std::pointer_traits<Foo>::pointer, Foo>::value, "");
