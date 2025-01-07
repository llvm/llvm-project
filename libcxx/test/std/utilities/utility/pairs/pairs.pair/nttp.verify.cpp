//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-17

// <utility>

// LWG-3382 NTTP for pair and array:
// pair<T, U> is a structural type ([temp.param]) if T and U are both structural types.

#include <utility>

#include <functional>
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
template <class T, class U, std::pair<T, U> P>
struct test {};

using A = test<int, int, std::pair{0, 1}>;
using B = test<int&, int&, std::make_pair(std::ref(i), std::ref(i))>;
using C = test<const int&, const int&, std::make_pair(std::cref(i), std::cref(i))>;
using D = test<LiteralType, LiteralType, std::pair<LiteralType, LiteralType>{}>;
using E = test<int*, int*, std::pair<int*, int*>{&i, &i}>;
using F = test<NotALiteral&, NotALiteral&, std::make_pair(std::ref(not_a_literal), std::ref(not_a_literal))>;

using G = test<int&&, int&&, std::pair<int&&, int&&>{std::move(i), std::move(i)}>;
// expected-error@*:* {{type 'std::pair<int &&, int &&>' of non-type template parameter is not a structural type}}

using H = test<NotALiteral, NotALiteral, std::pair<NotALiteral, NotALiteral>{}>;
// expected-error@*:* {{non-type template parameter has non-literal type 'std::pair<NotALiteral, NotALiteral>'}}

using I = test<std::string, std::string, std::pair<std::string, std::string>{}>;
// expected-error-re@*:* {{type 'std::pair<{{(std::)?}}string, {{(std::)?}}string>' {{(\(aka 'pair<basic_string<char>, basic_string<char>>'\) )?}}of non-type template parameter is not a structural type}}
} // namespace test_full_type

namespace test_ctad {
template <std::pair P>
struct test {};

using A = test<std::pair{2, 3}>;
using B = test<std::make_pair(std::ref(i), std::ref(i))>;
using C = test<std::make_pair(std::cref(i), std::cref(i))>;
using D = test<std::pair<LiteralType, LiteralType>{}>;
using E = test<std::pair<int*, int*>{&i, &i}>;
using F = test<std::make_pair(std::ref(not_a_literal), std::ref(not_a_literal))>;

using G = test<std::pair<int&&, int&&>{std::move(i), std::move(i)}>;
// expected-error@-1 {{type 'std::pair<int &&, int &&>' of non-type template parameter is not a structural type}}

using H = test<std::pair<NotALiteral, NotALiteral>{}>;
// expected-error@-1 {{non-type template parameter has non-literal type 'std::pair<NotALiteral, NotALiteral>'}}

using I = test<std::pair<std::string, std::string>{}>;
// expected-error@-1 {{type 'std::pair<string, string>' (aka 'std::pair<std::string, std::string>') of non-type template parameter is not a structural type}}
} // namespace test_ctad

namespace test_auto {
template <auto P>
struct test {};

using A = test<std::pair{4, 5}>;
using B = test<std::make_pair(std::ref(i), std::ref(i))>;
using C = test<std::make_pair(std::cref(i), std::cref(i))>;
using D = test<std::pair<LiteralType, LiteralType>{}>;
using E = test<std::pair<int*, int*>{&i, &i}>;
using F = test<std::make_pair(std::ref(not_a_literal), std::ref(not_a_literal))>;

using G = test<std::pair<int&&, int&&>{std::move(i), std::move(i)}>;
// expected-error@-1 {{type 'std::pair<int &&, int &&>' of non-type template parameter is not a structural type}}

using H = test<std::pair<NotALiteral, NotALiteral>{}>;
// expected-error@-1 {{non-type template parameter has non-literal type 'std::pair<NotALiteral, NotALiteral>'}}

using I = test<std::pair<std::string, std::string>{}>;
// expected-error@-1 {{type 'std::pair<std::string, std::string>' (aka 'pair<basic_string<char>, basic_string<char>>') of non-type template parameter is not a structural type}}
} // namespace test_auto
