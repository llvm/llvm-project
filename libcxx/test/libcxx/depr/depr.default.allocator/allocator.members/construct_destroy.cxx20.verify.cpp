//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// template <class... Args> void construct(pointer p, Args&&... args);
// void destroy(pointer p);

// Removed in C++20.

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <memory>

struct A {
  A();
  A(const A&);
  explicit A(int);
  A(int, int*);
};

class move_only {
  move_only(const move_only&)            = delete;
  move_only& operator=(const move_only&) = delete;

public:
  move_only();
  move_only(move_only&&);
  move_only& operator=(move_only&&);
};

void test(std::allocator<A>& a, A* ap) {
  a.construct(ap);             // expected-error {{no member}}
  a.destroy(ap);               // expected-error {{no member}}
  a.construct(ap, A());        // expected-error {{no member}}
  a.destroy(ap);               // expected-error {{no member}}
  a.construct(ap, 5);          // expected-error {{no member}}
  a.destroy(ap);               // expected-error {{no member}}
  a.construct(ap, 5, (int*)0); // expected-error {{no member}}
  a.destroy(ap);               // expected-error {{no member}}
}

void test(std::allocator<move_only>& a, move_only* ap) {
  a.construct(ap);              // expected-error {{no member}}
  a.destroy(ap);                // expected-error {{no member}}
  a.construct(ap, move_only()); // expected-error {{no member}}
  a.destroy(ap);                // expected-error {{no member}}
}
