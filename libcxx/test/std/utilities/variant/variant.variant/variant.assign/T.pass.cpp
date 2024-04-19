//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// template <class T>
// variant& operator=(T&&) noexcept(see below);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
#include <memory>

#include "test_macros.h"
#include "variant_test_helpers.h"

namespace MetaHelpers {

struct Dummy {
  Dummy() = default;
};

struct ThrowsCtorT {
  ThrowsCtorT(int) noexcept(false) {}
  ThrowsCtorT &operator=(int) noexcept { return *this; }
};

struct ThrowsAssignT {
  ThrowsAssignT(int) noexcept {}
  ThrowsAssignT &operator=(int) noexcept(false) { return *this; }
};

struct NoThrowT {
  NoThrowT(int) noexcept {}
  NoThrowT &operator=(int) noexcept { return *this; }
};

} // namespace MetaHelpers

namespace RuntimeHelpers {
#ifndef TEST_HAS_NO_EXCEPTIONS

struct ThrowsCtorT {
  int value;
  ThrowsCtorT() : value(0) {}
  ThrowsCtorT(int) noexcept(false) { throw 42; }
  ThrowsCtorT &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

struct MoveCrashes {
  int value;
  MoveCrashes(int v = 0) noexcept : value{v} {}
  MoveCrashes(MoveCrashes &&) noexcept { assert(false); }
  MoveCrashes &operator=(MoveCrashes &&) noexcept { assert(false); return *this; }
  MoveCrashes &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

struct ThrowsCtorTandMove {
  int value;
  ThrowsCtorTandMove() : value(0) {}
  ThrowsCtorTandMove(int) noexcept(false) { throw 42; }
  ThrowsCtorTandMove(ThrowsCtorTandMove &&) noexcept(false) { assert(false); }
  ThrowsCtorTandMove &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

struct ThrowsAssignT {
  int value;
  ThrowsAssignT() : value(0) {}
  ThrowsAssignT(int v) noexcept : value(v) {}
  ThrowsAssignT &operator=(int) noexcept(false) { throw 42; }
};

struct NoThrowT {
  int value;
  NoThrowT() : value(0) {}
  NoThrowT(int v) noexcept : value(v) {}
  NoThrowT &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

#endif // !defined(TEST_HAS_NO_EXCEPTIONS)
} // namespace RuntimeHelpers

void test_T_assignment_noexcept() {
  using namespace MetaHelpers;
  {
    using V = std::variant<Dummy, NoThrowT>;
    static_assert(std::is_nothrow_assignable<V, int>::value, "");
  }
  {
    using V = std::variant<Dummy, ThrowsCtorT>;
    static_assert(!std::is_nothrow_assignable<V, int>::value, "");
  }
  {
    using V = std::variant<Dummy, ThrowsAssignT>;
    static_assert(!std::is_nothrow_assignable<V, int>::value, "");
  }
}

void test_T_assignment_sfinae() {
  {
    using V = std::variant<long, long long>;
    static_assert(!std::is_assignable<V, int>::value, "ambiguous");
  }
  {
    using V = std::variant<std::string, std::string>;
    static_assert(!std::is_assignable<V, const char *>::value, "ambiguous");
  }
  {
    using V = std::variant<std::string, void *>;
    static_assert(!std::is_assignable<V, int>::value, "no matching operator=");
  }
  {
    using V = std::variant<std::string, float>;
    static_assert(!std::is_assignable<V, int>::value, "no matching operator=");
  }
  {
    using V = std::variant<std::unique_ptr<int>, bool>;
    static_assert(!std::is_assignable<V, std::unique_ptr<char>>::value,
                  "no explicit bool in operator=");
    struct X {
      operator void*();
    };
    static_assert(!std::is_assignable<V, X>::value, "no boolean conversion in operator=");
    static_assert(std::is_assignable<V, std::false_type>::value, "converted to bool in operator=");
  }
  {
    struct X {};
    struct Y {
      operator X();
    };
    using V = std::variant<X>;
    static_assert(std::is_assignable<V, Y>::value,
                  "regression on user-defined conversions in operator=");
  }
}

void test_T_assignment_basic() {
  {
    std::variant<int> v(43);
    v = 42;
    assert(v.index() == 0);
    assert(std::get<0>(v) == 42);
  }
  {
    std::variant<int, long> v(43l);
    v = 42;
    assert(v.index() == 0);
    assert(std::get<0>(v) == 42);
    v = 43l;
    assert(v.index() == 1);
    assert(std::get<1>(v) == 43);
  }
  {
    std::variant<unsigned, long> v;
    v = 42;
    assert(v.index() == 1);
    assert(std::get<1>(v) == 42);
    v = 43u;
    assert(v.index() == 0);
    assert(std::get<0>(v) == 43);
  }
  {
    std::variant<std::string, bool> v = true;
    v = "bar";
    assert(v.index() == 0);
    assert(std::get<0>(v) == "bar");
  }
  {
    std::variant<bool, std::unique_ptr<int>> v;
    v = nullptr;
    assert(v.index() == 1);
    assert(std::get<1>(v) == nullptr);
  }
}

void test_T_assignment_performs_construction() {
  using namespace RuntimeHelpers;
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V = std::variant<std::string, ThrowsCtorT>;
    V v(std::in_place_type<std::string>, "hello");
    try {
      v = 42;
      assert(false);
    } catch (...) { /* ... */
    }
    assert(v.index() == 0);
    assert(std::get<0>(v) == "hello");
  }
  {
    using V = std::variant<ThrowsAssignT, std::string>;
    V v(std::in_place_type<std::string>, "hello");
    v = 42;
    assert(v.index() == 0);
    assert(std::get<0>(v).value == 42);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

void test_T_assignment_performs_assignment() {
  using namespace RuntimeHelpers;
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V = std::variant<ThrowsCtorT>;
    V v;
    v = 42;
    assert(v.index() == 0);
    assert(std::get<0>(v).value == 42);
  }
  {
    using V = std::variant<ThrowsCtorT, std::string>;
    V v;
    v = 42;
    assert(v.index() == 0);
    assert(std::get<0>(v).value == 42);
  }
  {
    using V = std::variant<ThrowsAssignT>;
    V v(100);
    try {
      v = 42;
      assert(false);
    } catch (...) { /* ... */
    }
    assert(v.index() == 0);
    assert(std::get<0>(v).value == 100);
  }
  {
    using V = std::variant<std::string, ThrowsAssignT>;
    V v(100);
    try {
      v = 42;
      assert(false);
    } catch (...) { /* ... */
    }
    assert(v.index() == 1);
    assert(std::get<1>(v).value == 100);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

void test_T_assignment_vector_bool() {
  std::vector<bool> vec = {true};
  std::variant<bool, int> v;
  v = vec[0];
  assert(v.index() == 0);
  assert(std::get<0>(v) == true);
}

int main(int, char**) {
  test_T_assignment_basic();
  test_T_assignment_performs_construction();
  test_T_assignment_performs_assignment();
  test_T_assignment_noexcept();
  test_T_assignment_sfinae();
  test_T_assignment_vector_bool();

  return 0;
}
