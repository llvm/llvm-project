//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::function support for "blocks" when ARC is enabled

// UNSUPPORTED: c++03

// This test requires the Blocks runtime, which is (only?) available on Darwin
// out-of-the-box.
// REQUIRES: has-fblocks && has-fobjc-arc && darwin

// ADDITIONAL_COMPILE_FLAGS: -fblocks -fobjc-arc

#include <functional>

#include <cassert>
#include <cstddef>
#include <string>

struct Foo {
  Foo() = default;
  Foo(std::size_t (^bl)()) : f(bl) {}

  std::function<int()> f;
};

Foo Factory(std::size_t (^bl)()) {
  Foo result(bl);
  return result;
}

Foo Factory2() {
  auto hello = std::string("Hello world");
  return Factory(^() {
    return hello.size();
  });
}

Foo AssignmentFactory(std::size_t (^bl)()) {
  Foo result;
  result.f = bl;
  return result;
}

Foo AssignmentFactory2() {
  auto hello = std::string("Hello world");
  return AssignmentFactory(^() {
    return hello.size();
  });
}

int main(int, char **) {
  // Case 1, works
  {
    auto hello = std::string("Hello world");
    auto f = AssignmentFactory(^() {
      return hello.size();
    });
    assert(f.f() == 11);
  }

  // Case 2, works
  {
    auto f = AssignmentFactory2();
    assert(f.f() == 11);
  }

  // Case 3, works
  {
    auto hello = std::string("Hello world");
    auto f = Factory(^() {
      return hello.size();
    });
    assert(f.f() == 11);
  }

  // Case 4, used to crash under ARC
  {
    auto f = Factory2();
    assert(f.f() == 11);
  }

  return 0;
}
