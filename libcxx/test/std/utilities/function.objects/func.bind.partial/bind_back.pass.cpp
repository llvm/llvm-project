//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <functional>

// template<class F, class... Args>
//   constexpr unspecified bind_back(F&& f, Args&&... args);

#include <functional>

#include <cassert>
#include <concepts>
#include <tuple>
#include <utility>

#include "callable_types.h"
#include "types.h"

constexpr void test_basic_bindings() {
  { // Bind arguments, call without arguments
    {
      auto f = std::bind_back(MakeTuple{});
      assert(f() == std::make_tuple());
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{});
      assert(f() == std::make_tuple(Elem<1>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{}, Elem<2>{});
      assert(f() == std::make_tuple(Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f() == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
  }

  { // Bind no arguments, call with arguments
    {
      auto f = std::bind_back(MakeTuple{});
      assert(f(Elem<1>{}) == std::make_tuple(Elem<1>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{});
      assert(f(Elem<1>{}, Elem<2>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{});
      assert(f(Elem<1>{}, Elem<2>{}, Elem<3>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
  }

  { // Bind arguments, call with arguments
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<10>{}, Elem<1>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{}, Elem<2>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<10>{}, Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<10>{}, Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }

    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<10>{}, Elem<11>{}, Elem<1>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{}, Elem<2>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<10>{}, Elem<11>{}, Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<10>{}, Elem<11>{}, Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
    {
      auto f = std::bind_back(MakeTuple{}, Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}, Elem<11>{}, Elem<12>{}) ==
             std::make_tuple(Elem<10>{}, Elem<11>{}, Elem<12>{}, Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
  }

  { // Basic tests with fundamental types
    int n         = 2;
    int m         = 1;
    int sum       = 0;
    auto add      = [](int x, int y) { return x + y; };
    auto add_n    = [](int a, int b, int c, int d, int e, int f) { return a + b + c + d + e + f; };
    auto add_ref  = [&](int x, int y) -> int& { return sum = x + y; };
    auto add_rref = [&](int x, int y) -> int&& { return std::move(sum = x + y); };

    auto a = std::bind_back(add, m, n);
    assert(a() == 3);

    auto b = std::bind_back(add_n, m, n, m, m, m, m);
    assert(b() == 7);

    auto c = std::bind_back(add_n, n, m);
    assert(c(1, 1, 1, 1) == 7);

    auto d = std::bind_back(add_ref, n, m);
    std::same_as<int&> decltype(auto) dresult(d());
    assert(dresult == 3);

    auto e = std::bind_back(add_rref, n, m);
    std::same_as<int&&> decltype(auto) eresult(e());
    assert(eresult == 3);

    auto f = std::bind_back(add, n);
    assert(f(3) == 5);

    auto g = std::bind_back(add, n, 1);
    assert(g() == 3);

    auto h = std::bind_back(add_n, 1, 1, 1);
    assert(h(2, 2, 2) == 9);

    auto i = std::bind_back(add_ref, n);
    std::same_as<int&> decltype(auto) iresult(i(5));
    assert(iresult == 7);

    auto j = std::bind_back(add_rref, m);
    std::same_as<int&&> decltype(auto) jresult(j(4));
    assert(jresult == 5);
  }
}

constexpr void test_edge_cases() {
  { // Make sure we don't treat std::reference_wrapper specially.
    auto sub = [](std::reference_wrapper<int> a, std::reference_wrapper<int> b) { return a.get() - b.get(); };

    int i  = 1;
    int j  = 2;
    auto f = std::bind_back(sub, std::ref(i));
    assert(f(std::ref(j)) == 1);
  }

  { // Make sure we can call a function that's a pointer to a member function.
    struct MemberFunction {
      constexpr int foo(int x, int y) { return x * y; }
    };

    MemberFunction value;
    auto fn = std::bind_back(&MemberFunction::foo, 2, 3);
    assert(fn(value) == 6);
  }

  { // Make sure we can call a function that's a pointer to a member object.
    struct MemberObject {
      int obj;
    };

    MemberObject value{.obj = 3};
    auto fn = std::bind_back(&MemberObject::obj);
    assert(fn(value) == 3);
  }
}

constexpr void test_passing_arguments() {
  { // Make sure that we copy the bound arguments into the unspecified-type.
    auto add = [](int x, int y) { return x + y; };
    int n    = 2;
    auto f   = std::bind_back(add, n, 1);
    n        = 100;
    assert(f() == 3);
  }

  { // Make sure we pass the bound arguments to the function object
    // with the right value category.
    {
      auto was_copied = [](CopyMoveInfo info) { return info.copy_kind == CopyMoveInfo::copy; };
      CopyMoveInfo info;
      auto f = std::bind_back(was_copied, info);
      assert(f());
    }

    {
      auto was_moved = [](CopyMoveInfo info) { return info.copy_kind == CopyMoveInfo::move; };
      CopyMoveInfo info;
      auto f = std::bind_back(was_moved, info);
      assert(std::move(f)());
    }
  }
}

constexpr void test_function_objects() {
  { // Make sure we call the correctly cv-ref qualified operator()
    // based on the value category of the bind_back unspecified-type.
    struct X {
      constexpr int operator()() & { return 1; }
      constexpr int operator()() const& { return 2; }
      constexpr int operator()() && { return 3; }
      constexpr int operator()() const&& { return 4; }
    };

    auto f  = std::bind_back(X{});
    using F = decltype(f);
    assert(static_cast<F&>(f)() == 1);
    assert(static_cast<const F&>(f)() == 2);
    assert(static_cast<F&&>(f)() == 3);
    assert(static_cast<const F&&>(f)() == 4);
  }

  // Make sure the `bind_back` unspecified-type does not model invocable
  // when the call would select a differently-qualified operator().
  //
  // For example, if the call to `operator()() &` is ill-formed, the call to the unspecified-type
  // should be ill-formed and not fall back to the `operator()() const&` overload.
  { // Make sure we delete the & overload when the underlying call isn't valid.
    {
      struct X {
        void operator()() & = delete;
        void operator()() const&;
        void operator()() &&;
        void operator()() const&&;
      };

      using F = decltype(std::bind_back(X{}));
      static_assert(!std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(std::invocable<const F>);
    }

    // There's no way to make sure we delete the const& overload when the underlying call isn't valid,
    // so we can't check this one.

    { // Make sure we delete the && overload when the underlying call isn't valid.
      struct X {
        void operator()() &;
        void operator()() const&;
        void operator()() && = delete;
        void operator()() const&&;
      };

      using F = decltype(std::bind_back(X{}));
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(!std::invocable<F>);
      static_assert(std::invocable<const F>);
    }

    { // Make sure we delete the const&& overload when the underlying call isn't valid.
      struct X {
        void operator()() &;
        void operator()() const&;
        void operator()() &&;
        void operator()() const&& = delete;
      };

      using F = decltype(std::bind_back(X{}));
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(!std::invocable<const F>);
    }
  }

  { // Extra value category tests
    struct X {};

    {
      struct Y {
        void operator()(X&&) const&;
        void operator()(X&&) && = delete;
      };

      using F = decltype(std::bind_back(Y{}));
      static_assert(std::invocable<F&, X>);
      static_assert(!std::invocable<F, X>);
    }

    {
      struct Y {
        void operator()(const X&) const;
        void operator()(X&&) const = delete;
      };

      using F = decltype(std::bind_back(Y{}, X{}));
      static_assert(std::invocable<F&>);
      static_assert(!std::invocable<F>);
    }
  }
}

constexpr void test_return_type() {
  {   // Test properties of the constructor of the unspecified-type returned by bind_back.
    { // Test move constructor when function is move only.
      MoveOnlyCallable<bool> value(true);
      auto f = std::bind_back(std::move(value), 1);
      assert(f());
      assert(f(1, 2, 3));

      auto f1 = std::move(f);
      assert(!f());
      assert(f1());
      assert(f1(1, 2, 3));

      using F = decltype(f);
      static_assert(std::is_move_constructible<F>::value);
      static_assert(!std::is_copy_constructible<F>::value);
      static_assert(!std::is_move_assignable<F>::value);
      static_assert(!std::is_copy_assignable<F>::value);
    }

    { // Test move constructor when function is copyable but not assignable.
      CopyCallable<bool> value(true);
      auto f = std::bind_back(value, 1);
      assert(f());
      assert(f(1, 2, 3));

      auto f1 = std::move(f);
      assert(!f());
      assert(f1());
      assert(f1(1, 2, 3));

      auto f2 = std::bind_back(std::move(value), 1);
      assert(f1());
      assert(f2());
      assert(f2(1, 2, 3));

      using F = decltype(f);
      static_assert(std::is_move_constructible<F>::value);
      static_assert(std::is_copy_constructible<F>::value);
      static_assert(!std::is_move_assignable<F>::value);
      static_assert(!std::is_copy_assignable<F>::value);
    }

    { // Test constructors when function is copy assignable.
      using F = decltype(std::bind_back(std::declval<CopyAssignableWrapper&>(), 1));
      static_assert(std::is_move_constructible<F>::value);
      static_assert(std::is_copy_constructible<F>::value);
      static_assert(std::is_move_assignable<F>::value);
      static_assert(std::is_copy_assignable<F>::value);
    }

    { // Test constructors when function is move assignable only.
      using F = decltype(std::bind_back(std::declval<MoveAssignableWrapper>(), 1));
      static_assert(std::is_move_constructible<F>::value);
      static_assert(!std::is_copy_constructible<F>::value);
      static_assert(std::is_move_assignable<F>::value);
      static_assert(!std::is_copy_assignable<F>::value);
    }
  }

  { // Make sure bind_back's unspecified type's operator() is SFINAE-friendly.
    using F = decltype(std::bind_back(std::declval<int (*)(int, int)>(), 1));
    static_assert(!std::is_invocable<F>::value);
    static_assert(std::is_invocable<F, int>::value);
    static_assert(!std::is_invocable<F, void*>::value);
    static_assert(!std::is_invocable<F, int, int>::value);
  }
}

constexpr bool test() {
  test_basic_bindings();
  test_edge_cases();
  test_passing_arguments();
  test_function_objects();
  test_return_type();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
