//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <array>

// LWG-3382 NTTP for pair and array:
// array<T, N> is a structural type ([temp.param]) if T is a structural type.

#include <array>

#include <cstddef>
#include <string>

struct LiteralBase {};
struct LiteralNSDM {};

struct LiteralType : LiteralBase {
  LiteralNSDM nsdm;
};

struct NotALiteral {
  NotALiteral() {}
};

int i;
NotALiteral not_a_literal;

namespace test_full_type {
template <class T, std::size_t S, std::array<T, S> A>
struct test {};

using A = test<int, 2, std::array{2, 3}>;
using B = test<LiteralType, 0, std::array<LiteralType, 0>{}>;
using C = test<int*, 1, std::array<int*, 1>{&i}>;
using D = test<NotALiteral*, 1, std::array<NotALiteral*, 1>{&not_a_literal}>;

using E = test<NotALiteral, 1, std::array<NotALiteral, 1>{}>;
// expected-error-re@*:* {{non-type template parameter has non-literal type 'std::array<NotALiteral, 1U{{L{0,2}.*}}>'}}

using F = test<std::string, 2, std::array<std::string, 2>{}>;
// expected-error-re@*:* {{type {{.+}} of non-type template parameter is not a structural type}}
} // namespace test_full_type

namespace test_ctad {
template <std::array A>
struct test {};

using A = test<std::array{2, 3}>;
using B = test<std::array<LiteralType, 0>{}>;
using C = test<std::array<int*, 1>{&i}>;
using D = test<std::array<NotALiteral*, 1>{&not_a_literal}>;

using E = test<std::array<NotALiteral, 1>{}>;
// expected-error@-1 {{non-type template parameter has non-literal type 'std::array<NotALiteral, 1>'}}

using F = test<std::array<std::string, 2>{}>;
// expected-error-re@-1 {{type 'std::array<{{(std::)?}}string, 2>'{{( \(aka 'std::array<std::string, 2>'\))?}} of non-type template parameter is not a structural type}}
} // namespace test_ctad

namespace test_auto {
template <auto A>
struct test {};

using A = test<std::array{2, 3}>;
using B = test<std::array<LiteralType, 0>{}>;
using C = test<std::array<int*, 1>{&i}>;
using D = test<std::array<NotALiteral*, 1>{&not_a_literal}>;

using E = test<std::array<NotALiteral, 1>{}>;
// expected-error@-1 {{non-type template parameter has non-literal type 'std::array<NotALiteral, 1>'}}

using F = test<std::array<std::string, 2>{}>;
// expected-error-re@-1 {{type {{.+}} of non-type template parameter is not a structural type}}
} // namespace test_auto
