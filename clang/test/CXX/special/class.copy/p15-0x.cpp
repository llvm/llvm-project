// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// expected-no-diagnostics

namespace PR10622 {
  struct foo {
    const int first;
    foo(const foo&) = default;
  };
  void find_or_insert(const foo& __obj) {
    foo x(__obj);
  }

  struct bar : foo {
    bar(const bar&) = default;
  };
  void test_bar(const bar &obj) {
    bar obj2(obj);
  }
}

namespace PR11418 {
  template<typename T>
  T may_throw() {
    return T();
  }

  template<typename T> T &&declval() noexcept;

  struct NonPOD {
    NonPOD();
    NonPOD(const NonPOD &) noexcept;
    NonPOD(NonPOD &&) noexcept;
  };

  struct X {
    NonPOD np = may_throw<NonPOD>();
  };

  static_assert(noexcept(declval<X>()), "noexcept isn't working at all");
  static_assert(noexcept(X(declval<X&>())), "copy constructor can't throw");
  static_assert(noexcept(X(declval<X>())), "move constructor can't throw");
}
