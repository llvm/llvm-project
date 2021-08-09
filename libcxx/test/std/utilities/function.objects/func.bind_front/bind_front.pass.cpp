//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// functional

// template <class F, class... Args>
// constexpr unspecified bind_front(F&&, Args&&...);

#include <functional>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>

#include "callable_types.h"
#include "test_macros.h"

struct CopyMoveInfo {
  enum { none, copy, move } copy_kind;

  constexpr CopyMoveInfo() : copy_kind(none) {}
  constexpr CopyMoveInfo(CopyMoveInfo const&) : copy_kind(copy) {}
  constexpr CopyMoveInfo(CopyMoveInfo&&) : copy_kind(move) {}
};

template <class ...Args>
struct is_bind_frontable {
  template <class ...LocalArgs>
  static auto test(int)
      -> decltype((void)std::bind_front(std::declval<LocalArgs>()...), std::true_type());

  template <class...>
  static std::false_type test(...);

  static constexpr bool value = decltype(test<Args...>(0))::value;
};

struct NotCopyMove {
  NotCopyMove() = delete;
  NotCopyMove(const NotCopyMove&) = delete;
  NotCopyMove(NotCopyMove&&) = delete;
  template <class ...Args>
  void operator()(Args&& ...) const { }
};

struct NonConstCopyConstructible {
  explicit NonConstCopyConstructible() {}
  NonConstCopyConstructible(NonConstCopyConstructible&) {}
};

struct MoveConstructible {
  explicit MoveConstructible() {}
  MoveConstructible(MoveConstructible&&) {}
};

struct MakeTuple {
  template <class ...Args>
  constexpr auto operator()(Args&& ...args) const {
    return std::make_tuple(std::forward<Args>(args)...);
  }
};

template <int X>
struct Elem {
  template <int Y>
  constexpr bool operator==(Elem<Y> const&) const
  { return X == Y; }
};

constexpr bool test() {
  // Bind arguments, call without arguments
  {
    {
      auto f = std::bind_front(MakeTuple{});
      assert(f() == std::make_tuple());
    }
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{});
      assert(f() == std::make_tuple(Elem<1>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{}, Elem<2>{});
      assert(f() == std::make_tuple(Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f() == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
  }

  // Bind no arguments, call with arguments
  {
    {
      auto f = std::bind_front(MakeTuple{});
      assert(f(Elem<1>{}) == std::make_tuple(Elem<1>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{});
      assert(f(Elem<1>{}, Elem<2>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{});
      assert(f(Elem<1>{}, Elem<2>{}, Elem<3>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}));
    }
  }

  // Bind arguments, call with arguments
  {
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<1>{}, Elem<10>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{}, Elem<2>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<10>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}, Elem<10>{}));
    }

    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<1>{}, Elem<10>{}, Elem<11>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{}, Elem<2>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<10>{}, Elem<11>{}));
    }
    {
      auto f = std::bind_front(MakeTuple{}, Elem<1>{}, Elem<2>{}, Elem<3>{});
      assert(f(Elem<10>{}, Elem<11>{}) == std::make_tuple(Elem<1>{}, Elem<2>{}, Elem<3>{}, Elem<10>{}, Elem<11>{}));
    }
  }

  // Basic tests with fundamental types
  {
    int n = 2;
    int m = 1;
    auto add = [](int x, int y) { return x + y; };
    auto addN = [](int a, int b, int c, int d, int e, int f) {
      return a + b + c + d + e + f;
    };

    auto a = std::bind_front(add, m, n);
    assert(a() == 3);

    auto b = std::bind_front(addN, m, n, m, m, m, m);
    assert(b() == 7);

    auto c = std::bind_front(addN, n, m);
    assert(c(1, 1, 1, 1) == 7);

    auto f = std::bind_front(add, n);
    assert(f(3) == 5);

    auto g = std::bind_front(add, n, 1);
    assert(g() == 3);

    auto h = std::bind_front(addN, 1, 1, 1);
    assert(h(2, 2, 2) == 9);
  }

  // Make sure we don't treat std::reference_wrapper specially.
  {
    auto add = [](std::reference_wrapper<int> a, std::reference_wrapper<int> b) {
      return a.get() + b.get();
    };
    int i = 1, j = 2;
    auto f = std::bind_front(add, std::ref(i));
    assert(f(std::ref(j)) == 3);
  }

  // Make sure we can call a function that's a pointer to a member function.
  {
    struct MemberFunction {
      constexpr bool foo(int, int) { return true; }
    };
    MemberFunction value;
    auto fn = std::bind_front(&MemberFunction::foo, value, 0);
    assert(fn(0));
  }

  // Make sure that we copy the bound arguments into the unspecified-type.
  {
    auto add = [](int x, int y) { return x + y; };
    int n = 2;
    auto i = std::bind_front(add, n, 1);
    n = 100;
    assert(i() == 3);
  }

  // Make sure we pass the bound arguments to the function object
  // with the right value category.
  {
    {
      auto wasCopied = [](CopyMoveInfo info) {
        return info.copy_kind == CopyMoveInfo::copy;
      };
      CopyMoveInfo info;
      auto copied = std::bind_front(wasCopied, info);
      assert(copied());
    }

    {
      auto wasMoved = [](CopyMoveInfo info) {
        return info.copy_kind == CopyMoveInfo::move;
      };
      CopyMoveInfo info;
      auto moved = std::bind_front(wasMoved, info);
      assert(std::move(moved)());
    }
  }

  // Make sure we call the correctly cv-ref qualified operator() based on the
  // value category of the bind_front unspecified-type.
  {
    struct F {
      constexpr int operator()() & { return 1; }
      constexpr int operator()() const& { return 2; }
      constexpr int operator()() && { return 3; }
      constexpr int operator()() const&& { return 4; }
    };
    auto x = std::bind_front(F{});
    using X = decltype(x);
    assert(static_cast<X&>(x)() == 1);
    assert(static_cast<X const&>(x)() == 2);
    assert(static_cast<X&&>(x)() == 3);
    assert(static_cast<X const&&>(x)() == 4);
  }

  // Make sure the bind_front unspecified-type is NOT invocable when the call would select a
  // differently-qualified operator().
  //
  // For example, if the call to `operator()() &` is ill-formed, the call to the unspecified-type
  // should be ill-formed and not fall back to the `operator()() const&` overload.
  {
    // Make sure we delete the & overload when the underlying call isn't valid
    {
      struct F {
        void operator()() & = delete;
        void operator()() const&;
        void operator()() &&;
        void operator()() const&&;
      };
      using X = decltype(std::bind_front(F{}));
      static_assert(!std::is_invocable_v<X&>);
      static_assert( std::is_invocable_v<X const&>);
      static_assert( std::is_invocable_v<X>);
      static_assert( std::is_invocable_v<X const>);
    }

    // There's no way to make sure we delete the const& overload when the underlying call isn't valid,
    // so we can't check this one.

    // Make sure we delete the && overload when the underlying call isn't valid
    {
      struct F {
        void operator()() &;
        void operator()() const&;
        void operator()() && = delete;
        void operator()() const&&;
      };
      using X = decltype(std::bind_front(F{}));
      static_assert( std::is_invocable_v<X&>);
      static_assert( std::is_invocable_v<X const&>);
      static_assert(!std::is_invocable_v<X>);
      static_assert( std::is_invocable_v<X const>);
    }

    // Make sure we delete the const&& overload when the underlying call isn't valid
    {
      struct F {
        void operator()() &;
        void operator()() const&;
        void operator()() &&;
        void operator()() const&& = delete;
      };
      using X = decltype(std::bind_front(F{}));
      static_assert( std::is_invocable_v<X&>);
      static_assert( std::is_invocable_v<X const&>);
      static_assert( std::is_invocable_v<X>);
      static_assert(!std::is_invocable_v<X const>);
    }
  }

  // Some examples by Tim Song
  {
    {
      struct T { };
      struct F {
        void operator()(T&&) const &;
        void operator()(T&&) && = delete;
      };
      using X = decltype(std::bind_front(F{}));
      static_assert(!std::is_invocable_v<X, T>);
    }

    {
      struct T { };
      struct F {
        void operator()(T const&) const;
        void operator()(T&&) const = delete;
      };
      using X = decltype(std::bind_front(F{}, T{}));
      static_assert(!std::is_invocable_v<X>);
    }
  }

  // Test properties of the constructor of the unspecified-type returned by bind_front.
  {
    {
      MoveOnlyCallable value(true);
      using RetT = decltype(std::bind_front(std::move(value), 1));

      static_assert( std::is_move_constructible<RetT>::value);
      static_assert(!std::is_copy_constructible<RetT>::value);
      static_assert(!std::is_move_assignable<RetT>::value);
      static_assert(!std::is_copy_assignable<RetT>::value);

      auto ret = std::bind_front(std::move(value), 1);
      assert(ret());
      assert(ret(1, 2, 3));

      auto ret1 = std::move(ret);
      assert(!ret());
      assert(ret1());
      assert(ret1(1, 2, 3));
    }
    {
      CopyCallable value(true);
      using RetT = decltype(std::bind_front(value, 1));

      static_assert( std::is_move_constructible<RetT>::value);
      static_assert( std::is_copy_constructible<RetT>::value);
      static_assert(!std::is_move_assignable<RetT>::value);
      static_assert(!std::is_copy_assignable<RetT>::value);

      auto ret = std::bind_front(value, 1);
      assert(ret());
      assert(ret(1, 2, 3));

      auto ret1 = std::move(ret);
      assert(ret1());
      assert(ret1(1, 2, 3));

      auto ret2 = std::bind_front(std::move(value), 1);
      assert(!ret());
      assert(ret2());
      assert(ret2(1, 2, 3));
    }
    {
      CopyAssignableWrapper value(true);
      using RetT = decltype(std::bind_front(value, 1));

      static_assert(std::is_move_constructible<RetT>::value);
      static_assert(std::is_copy_constructible<RetT>::value);
      static_assert(std::is_move_assignable<RetT>::value);
      static_assert(std::is_copy_assignable<RetT>::value);
    }
    {
      MoveAssignableWrapper value(true);
      using RetT = decltype(std::bind_front(std::move(value), 1));

      static_assert( std::is_move_constructible<RetT>::value);
      static_assert(!std::is_copy_constructible<RetT>::value);
      static_assert( std::is_move_assignable<RetT>::value);
      static_assert(!std::is_copy_assignable<RetT>::value);
    }
  }

  // Make sure bind_front is SFINAE friendly
  {
    using T = decltype(std::bind_front(std::declval<int(*)(int, int)>(), 1));
    static_assert(!std::is_invocable<T>::value);
    static_assert( std::is_invocable<T, int>::value);
    static_assert(!std::is_invocable<T, void*>::value);
    static_assert(!std::is_invocable<T, int, int>::value);

    static_assert(!std::is_constructible_v<NotCopyMove, NotCopyMove&>);
    static_assert(!std::is_move_constructible_v<NotCopyMove>);
    static_assert(!is_bind_frontable<NotCopyMove>::value);
    static_assert(!is_bind_frontable<NotCopyMove&>::value);

    auto takeAnything = [](auto&& ...) { };
    static_assert(!std::is_constructible_v<MoveConstructible, MoveConstructible&>);
    static_assert( std::is_move_constructible_v<MoveConstructible>);
    static_assert( is_bind_frontable<decltype(takeAnything), MoveConstructible>::value);
    static_assert(!is_bind_frontable<decltype(takeAnything), MoveConstructible&>::value);

    static_assert( std::is_constructible_v<NonConstCopyConstructible, NonConstCopyConstructible&>);
    static_assert(!std::is_move_constructible_v<NonConstCopyConstructible>);
    static_assert(!is_bind_frontable<decltype(takeAnything), NonConstCopyConstructible&>::value);
    static_assert(!is_bind_frontable<decltype(takeAnything), NonConstCopyConstructible>::value);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
