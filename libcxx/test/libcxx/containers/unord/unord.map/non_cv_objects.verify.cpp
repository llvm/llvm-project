//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that we can only form maps of object types.

#include <unordered_map>

struct S {};

namespace std {
template <>
struct hash<S> {
  using argument_type = S;
  using result_type   = size_t;

  size_t operator()(S) const;
};

template <>
struct hash<S const> {
  using argument_type = S;
  using result_type   = size_t;

  size_t operator()(S) const;
};

template <>
struct hash<S volatile> {
  using argument_type = S;
  using result_type   = size_t;

  size_t operator()(S const volatile&) const;
};
} // namespace std

std::unordered_map<S const, int> K1;
std::unordered_map<int, int const> M1;
// TODO(#106635): turn this into a compile-time error

std::unordered_map<S volatile, int> K2;
std::unordered_map<int, int volatile> M2;
// TODO(#106635): turn this into a compile-time error

std::unordered_map<int&, int> K3;
std::unordered_map<int, int&> M3; // TODO(#106635): turn this into a compile-time error
// expected-error@*:* 1 {{'std::unordered_map' cannot hold references}}

std::unordered_map<int&&, int> K4;
std::unordered_map<int, int&&> M4; // TODO(#106635): turn this into a compile-time error
// expected-error@*:*{{'std::unordered_map' cannot hold references}}

std::unordered_map<int(), int> K5;
std::unordered_map<int(int), int> K6;
std::unordered_map<int(int, int), int> K7;
std::unordered_map<int, int()> M5;
std::unordered_map<int, int(int)> M6;
std::unordered_map<int, int(int, int)> M7;
// expected-error@*:* 6 {{'std::unordered_map' cannot hold functions}}

std::unordered_map<void, int> K8;
std::unordered_map<int, void> M8;
// expected-error@*:* 2 {{'std::unordered_map' cannot hold 'void'}}

std::unordered_map<int[], int> K9;
std::unordered_map<int, int[]> M9; // TODO(#106635): turn this into a compile-time error
// expected-error@*:*{{'std::unordered_map' cannot hold C arrays of an unknown size}}

std::unordered_map<int[2], int> K10;
std::unordered_map<int, int[2]> M10; // TODO(#106635): turn this into a compile-time error
// expected-error@*:*{{'std::unordered_map' cannot hold C arrays before C++20}}
