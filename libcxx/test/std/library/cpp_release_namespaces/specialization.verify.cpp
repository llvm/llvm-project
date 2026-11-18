//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// This documents Clang's behavior, which may or may not be conforming (qualified lookup of the declarator-id of an
// explicit specialization through a using-directive). Other compilers may behave differently.
// UNSUPPORTED: gcc

// PROTOTYPE: P3866R0 (V2: An Evolution Path for the Standard Library)
//
// The paper recommends using the release namespace by default. Can users specialize customization points like
// std::hash through it?

#include <cstddef>
#include <functional>
#include <vector>

namespace cpp = std::cpp29;

struct X {};
struct Y {};
struct Z {};

// Specializing through a qualified name works with Clang, both directly and through the namespace alias.
template <>
struct std::cpp29::hash<X> {
  std::size_t operator()(X) const { return 0; }
};

template <>
struct cpp::hash<Y> {
  std::size_t operator()(Y) const { return 0; }
};

// But reopening the release namespace doesn't, since the entity isn't actually declared there.
namespace std::cpp29 {
template <>
struct hash<Z> { // expected-error {{class template specialization of 'hash' not in a namespace enclosing}}
  std::size_t operator()(Z) const { return 0; }
};
} // namespace std::cpp29

void f() {
  (void)std::hash<X>()(X());
  (void)std::hash<Y>()(Y());
}
