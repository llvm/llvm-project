//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Make sure the rvalue overload of operator>> isn't part of the overload set
// when the type is not input streamable from a lvalue stream.

#include <istream>
#include <utility>

struct Foo { };

using X = decltype(std::declval<std::istream>() >> std::declval<Foo&>()); // expected-error {{invalid operands to binary expression}}
using Y = decltype(std::declval<std::istream>() >> std::declval<Foo>()); // expected-error {{invalid operands to binary expression}}
