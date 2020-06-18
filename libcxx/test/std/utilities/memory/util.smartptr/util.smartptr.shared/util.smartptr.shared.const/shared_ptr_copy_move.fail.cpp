//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y> shared_ptr(const shared_ptr<Y>& r);

#include <memory>
#include <type_traits>

struct A {
  int x = 42;
};

struct ADel : std::default_delete<A> {
  typedef A* pointer;
};

int main(int, char**) {
  static_assert(!(std::is_convertible<A, int>::value), "");

  {
    std::shared_ptr<A> pA;
    std::shared_ptr<int> pi(pA); // expected-error {{no matching constructor for initialization of 'std::shared_ptr<int>'}}
  }
  {
    std::shared_ptr<A> pA;
    std::shared_ptr<int> pi(std::move(pA)); // expected-error {{no matching constructor for initialization of 'std::shared_ptr<int>'}}
  }
  {
    std::weak_ptr<A> pA;
    std::shared_ptr<int> pi(std::move(pA)); // expected-error {{no matching constructor for initialization of 'std::shared_ptr<int>'}}
  }

#if TEST_STD_VER > 14
  {
    std::unique_ptr<int, ADel> ui;
    std::shared_ptr<int> pi(std::move(ui)); // expected-error {{no matching constructor for initialization of 'std::shared_ptr<int>'}}
  }
#endif

  return 0;
}
