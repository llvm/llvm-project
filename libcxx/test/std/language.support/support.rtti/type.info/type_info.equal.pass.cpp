//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// class type_info
//
//  bool operator==(const type_info& rhs) const noexcept; // constexpr since C++23

// UNSUPPORTED: no-rtti

// When we build for Windows on top of the VC runtime, `typeinfo::operator==` may not
// be `constexpr` (depending on the version of the VC runtime). So this test can fail.
// UNSUPPORTED: target={{.+}}-windows-msvc && !libcpp-no-vcruntime

#include <typeinfo>
#include <cassert>

#include "test_macros.h"

struct Base {
  virtual void func() {}
};
struct Derived : Base {
  virtual void func() {}
};

TEST_CONSTEXPR_CXX23 bool test() {
  // Test when storing typeid() in a const ref
  {
    std::type_info const& t1 = typeid(int);
    std::type_info const& t2 = typeid(long);
    assert(t1 == t1);
    assert(t2 == t2);
    assert(t1 != t2);
  }

  // Test when using `typeid()` directly
  {
    struct Foo { };
    struct Bar { };
    assert(typeid(Foo) == typeid(Foo));
    assert(typeid(Foo) != typeid(Bar));
  }

  // Test when using typeid(object) instead of typeid(type)
  {
    int x = 0, y = 0;
    long z = 0;
    assert(typeid(x) == typeid(y));
    assert(typeid(x) != typeid(z));
  }

  // Check with derived/base types
  {
    Derived derived;
    Base const& as_base = derived;
    assert(typeid(as_base) == typeid(Derived));
  }

  // Check noexcept-ness
  {
    std::type_info const& t1 = typeid(int); (void)t1;
    std::type_info const& t2 = typeid(long); (void)t2;
    ASSERT_NOEXCEPT(t1 == t2);
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
  return 0;
}
