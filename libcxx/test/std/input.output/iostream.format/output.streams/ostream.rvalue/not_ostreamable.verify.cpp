//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure the rvalue overload of operator<< isn't part of the overload set
// when the type is not output streamable into a lvalue stream.

#include <ostream>
#include <utility>

struct Foo { };

using X = decltype(std::declval<std::ostream>() << std::declval<Foo const&>()); // expected-error {{invalid operands to binary expression}}
