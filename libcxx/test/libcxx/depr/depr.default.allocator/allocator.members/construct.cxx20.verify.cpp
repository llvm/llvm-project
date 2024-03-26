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

// Removed in C++20.

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <memory>
#include <cassert>

int A_constructed = 0;

struct A {
  int data;
  A() { ++A_constructed; }

  A(const A&) { ++A_constructed; }

  explicit A(int) { ++A_constructed; }
  A(int, int*) { ++A_constructed; }

  ~A() { --A_constructed; }
};

int move_only_constructed = 0;

class move_only {
  move_only(const move_only&)            = delete;
  move_only& operator=(const move_only&) = delete;

public:
  move_only(move_only&&) { ++move_only_constructed; }
  move_only& operator=(move_only&&) { return *this; }

  move_only() { ++move_only_constructed; }
  ~move_only() { --move_only_constructed; }

public:
  int data; // unused other than to make sizeof(move_only) == sizeof(int).
            // but public to suppress "-Wunused-private-field"
};

int main(int, char**) {
  {
    std::allocator<A> a;
    A* ap = a.allocate(3);
    a.construct(ap);             // expected-error {{no member}}
    a.destroy(ap);               // expected-error {{no member}}
    a.construct(ap, A());        // expected-error {{no member}}
    a.destroy(ap);               // expected-error {{no member}}
    a.construct(ap, 5);          // expected-error {{no member}}
    a.destroy(ap);               // expected-error {{no member}}
    a.construct(ap, 5, (int*)0); // expected-error {{no member}}
    a.destroy(ap);               // expected-error {{no member}}
    a.deallocate(ap, 3);
  }
  {
    std::allocator<move_only> a;
    move_only* ap = a.allocate(3);
    a.construct(ap);              // expected-error {{no member}}
    a.destroy(ap);                // expected-error {{no member}}
    a.construct(ap, move_only()); // expected-error {{no member}}
    a.destroy(ap);                // expected-error {{no member}}
    a.deallocate(ap, 3);
  }
  return 0;
}
