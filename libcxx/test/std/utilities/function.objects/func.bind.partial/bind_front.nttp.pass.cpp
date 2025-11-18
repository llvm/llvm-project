//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <functional>

// template<auto f, class... Args>
//   constexpr unspecified bind_front(Args&&...);

#include <functional>

#include <cassert>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include "types.h"

constexpr void test_basic_bindings() {
  { // Bind arguments, call without arguments
    {
      auto f = std::bind_front<MakeTuple{}>();
      assert(f() == std::make_tuple());
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{});
      assert(f() == std::make_tuple(Elem<1>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{}, Elem<2>{});
      assert(f() == std::make_tuple(Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f() == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
  }

  { // Bind no arguments, call with arguments
    {
      auto f = std::bind_front<MakeTuple{}>();
      assert(f(Elem<1>{}) == std::make_tuple(Elem<1>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>();
      assert(f(Elem<1>{}, Elem<2>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>();
      assert(f(Elem<1>{}, Elem<2>{}, Elem<3>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
  }

  { // Bind arguments, call with arguments
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<1>{}, Elem<10>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{}, Elem<2>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<10>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}, Elem<10>{}));
    }

    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<1>{}, Elem<10>{}, Elem<11>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{}, Elem<2>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<10>{}, Elem<11>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}, Elem<10>{}, Elem<11>{}));
    }
    {
      auto f = std::bind_front<MakeTuple{}>(Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}, Elem<11>{}, Elem<12>{}) ==
             std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}, Elem<10>{}, Elem<11>{}, Elem<12>{}));
    }
  }

  { // Basic tests with fundamental types
    const int n = 2;
    const int m = 1;
    int o       = 0;

    auto add       = [](int x, int y) { return x + y; };
    auto add6      = [](int a, int b, int c, int d, int e, int f) { return a + b + c + d + e + f; };
    auto increment = [](int& x) { return ++x; };

    auto a = std::bind_front<add>(m, n);
    assert(a() == 3);

    auto b = std::bind_front<add6>(m, n, m, m, m, m);
    assert(b() == 7);

    auto c = std::bind_front<add6>(n, m);
    assert(c(1, 1, 1, 1) == 7);

    auto f = std::bind_front<add>(n);
    assert(f(3) == 5);

    auto g = std::bind_front<add>(n, 1);
    assert(g() == 3);

    auto h = std::bind_front<add6>(1, 1, 1);
    assert(h(2, 2, 2) == 9);

    auto i = std::bind_front<increment>();
    assert(i(o) == 1);
    assert(o == 1);

    auto j = std::bind_front<increment>(std::ref(o));
    assert(j() == 2);
    assert(o == 2);
  }
}

constexpr void test_edge_cases() {
  { // Make sure we don't treat std::reference_wrapper specially.
    auto sub = [](std::reference_wrapper<int> a, std::reference_wrapper<int> b) { return a.get() - b.get(); };

    int i  = 1;
    int j  = 2;
    auto f = std::bind_front<sub>(std::ref(i));
    assert(f(std::ref(j)) == -1);
  }

  { // Make sure we can call a function that's a pointer to a member function.
    struct MemberFunction {
      constexpr int mul(int x, int y) { return x * y; }
    };

    MemberFunction value;
    auto fn = std::bind_front<&MemberFunction::mul>(value, 2);
    assert(fn(3) == 6);
  }

  { // Make sure we can call a function that's a pointer to a member object.
    struct MemberObject {
      int obj;
    };

    MemberObject value{.obj = 3};
    auto fn1 = std::bind_front<&MemberObject::obj>();
    assert(fn1(value) == 3);
    auto fn2 = std::bind_front<&MemberObject::obj>(value);
    assert(fn2() == 3);
  }
}

constexpr void test_passing_arguments() {
  { // Make sure that we copy the bound arguments into the unspecified-type.
    int n  = 2;
    auto f = std::bind_front<[](int x, int y) { return x + y; }>(n, 1);
    n      = 100;
    assert(f() == 3);
  }

  { // Make sure we pass the bound arguments to the function object
    // with the right value category.
    {
      auto was_copied = [](CopyMoveInfo info) { return info.copy_kind == CopyMoveInfo::copy; };
      CopyMoveInfo info;
      auto f = std::bind_front<was_copied>(info);
      assert(f());
    }

    {
      auto was_moved = [](CopyMoveInfo info) { return info.copy_kind == CopyMoveInfo::move; };
      CopyMoveInfo info;
      auto f = std::bind_front<was_moved>(info);
      assert(std::move(f)());
    }
  }
}

constexpr void test_perfect_forwarding_call_wrapper() {
  { // Make sure we call the correctly cv-ref qualified operator()
    // based on the value category of the bind_front<NTTP> unspecified-type.
    struct X {
      constexpr int operator()() & { return 1; }
      constexpr int operator()() const& { return 2; }
      constexpr int operator()() && { return 3; }
      constexpr int operator()() const&& { return 4; }
    };

    auto f  = std::bind_front<X{}>();
    using F = decltype(f);
    assert(static_cast<F&>(f)() == 2);
    assert(static_cast<const F&>(f)() == 2);
    assert(static_cast<F&&>(f)() == 2);
    assert(static_cast<const F&&>(f)() == 2);
  }

  // Call to `bind_front<NTTP>` unspecified-type's operator() should always result in call to the const& overload of the underlying function object.
  {
    { // Make sure unspecified-type is still callable when we delete the & overload.
      struct X {
        int operator()() & = delete;
        int operator()() const&;
        int operator()() &&;
        int operator()() const&&;
      };

      using F = decltype(std::bind_front<X{}>());
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(std::invocable<const F>);
    }

    { // Make sure unspecified-type is not callable when we delete the const& overload.
      struct X {
        int operator()() &;
        int operator()() const& = delete;
        int operator()() &&;
        int operator()() const&&;
      };

      using F = decltype(std::bind_front<X{}>());
      static_assert(!std::invocable<F&>);
      static_assert(!std::invocable<const F&>);
      static_assert(!std::invocable<F>);
      static_assert(!std::invocable<const F>);
    }

    { // Make sure unspecified-type is still callable when we delete the && overload.
      struct X {
        int operator()() &;
        int operator()() const&;
        int operator()() && = delete;
        int operator()() const&&;
      };

      using F = decltype(std::bind_front<X{}>());
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(std::invocable<const F>);
    }

    { // Make sure unspecified-type is still callable when we delete the const&& overload.
      struct X {
        int operator()() &;
        int operator()() const&;
        int operator()() &&;
        int operator()() const&& = delete;
      };

      using F = decltype(std::bind_front<X{}>());
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(std::invocable<const F>);
    }
  }

  { // Test perfect forwarding when various overloads are available
    struct X {
      Tag<0> operator()(int&, char) const;
      Tag<1> operator()(const int&, char) const;
      Tag<2> operator()(int&&, char) const;
      Tag<3> operator()(const int&&, char) const;
    };

    using F = decltype(std::bind_front<X{}>(0));
    static_assert(std::same_as<std::invoke_result_t<F&, char>, Tag<0>>);
    static_assert(std::same_as<std::invoke_result_t<const F&, char>, Tag<1>>);
    static_assert(std::same_as<std::invoke_result_t<F, char>, Tag<2>>);
    static_assert(std::same_as<std::invoke_result_t<const F, char>, Tag<3>>);
  }

  { // Test perfect forwarding
    auto f = [](int& val) {
      val = 5;
      return 10;
    };

    auto bf = std::bind_front<f>();
    int val = 0;
    assert(bf(val) == 10);
    assert(val == 5);

    using BF = decltype(bf);
    static_assert(std::invocable<BF, int&>);
    static_assert(!std::invocable<BF, int>);
  }
}

constexpr void test_return_type() {
  { // Test constructors and assignment operators
    struct LeftShift {
      constexpr unsigned int operator()(unsigned int x, unsigned int y) const { return x << y; }
    };

    auto power_of_2 = std::bind_front<LeftShift{}>(1);
    assert(power_of_2(5) == 32U);
    assert(power_of_2(4) == 16U);

    auto moved = std::move(power_of_2);
    assert(moved(6) == 64);
    assert(moved(7) == 128);

    auto copied = power_of_2;
    assert(copied(3) == 8);
    assert(copied(2) == 4);

    moved = std::move(copied);
    assert(copied(1) == 2);
    assert(copied(0) == 1);

    copied = moved;
    assert(copied(8) == 256);
    assert(copied(9) == 512);
  }

  { // Make sure `bind_front<NTTP>` unspecified-type's operator() is SFINAE-friendly.
    using F = decltype(std::bind_front<[](int x, int y) { return x / y; }>(1));
    static_assert(!std::is_invocable<F>::value);
    static_assert(std::is_invocable<F, int>::value);
    static_assert(!std::is_invocable<F, void*>::value);
    static_assert(!std::is_invocable<F, int, int>::value);
  }

  { // Test noexceptness
    auto always_noexcept = std::bind_front<MaybeNoexceptFn<true>{}>();
    static_assert(noexcept(always_noexcept()));

    auto never_noexcept = std::bind_front<MaybeNoexceptFn<false>{}>();
    static_assert(!noexcept(never_noexcept()));
  }

  { // Test calling volatile wrapper -- we allow it as an extension
    using Fn = decltype(std::bind_front<std::integral_constant<int, 0>{}>());
    static_assert(std::invocable<volatile Fn&>);
    static_assert(std::invocable<const volatile Fn&>);
    static_assert(std::invocable<volatile Fn>);
    static_assert(std::invocable<const volatile Fn>);
  }
}

constexpr bool test() {
  test_basic_bindings();
  test_edge_cases();
  test_passing_arguments();
  test_perfect_forwarding_call_wrapper();
  test_return_type();

  return true;
}

int main(int, char**) {
  test();
  static_assert((test(), true));

  return 0;
}
