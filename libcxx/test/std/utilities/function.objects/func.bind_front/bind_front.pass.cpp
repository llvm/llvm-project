//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// functional

// template <class F, class... Args> constexpr unspecified bind_front(F&&, Args&&...);

#include <functional>

#include "callable_types.h"
#include "test_macros.h"

constexpr int add(int a, int b) { return a + b; }

constexpr int long_test(int a, int b, int c, int d, int e, int f) {
  return a + b + c + d + e + f;
}

struct Foo {
  int a;
  int b;
};

struct FooCall {
  constexpr Foo operator()(int a, int b) { return Foo{a, b}; }
};

struct S {
  constexpr bool operator()(int a) { return a == 1; }
};

struct CopyMoveInfo {
  enum { none, copy, move } copy_kind;

  constexpr CopyMoveInfo() : copy_kind(none) {}
  constexpr CopyMoveInfo(CopyMoveInfo const&) : copy_kind(copy) {}
  constexpr CopyMoveInfo(CopyMoveInfo&&) : copy_kind(move) {}
};

constexpr bool wasCopied(CopyMoveInfo info) {
  return info.copy_kind == CopyMoveInfo::copy;
}
constexpr bool wasMoved(CopyMoveInfo info) {
  return info.copy_kind == CopyMoveInfo::move;
}

constexpr void basic_tests() {
  int n = 2;
  int m = 1;

  auto a = std::bind_front(add, m, n);
  assert(a() == 3);

  auto b = std::bind_front(long_test, m, n, m, m, m, m);
  assert(b() == 7);

  auto c = std::bind_front(long_test, n, m);
  assert(c(1, 1, 1, 1) == 7);

  auto d = std::bind_front(S{}, m);
  assert(d());

  auto f = std::bind_front(add, n);
  assert(f(3) == 5);

  auto g = std::bind_front(add, n, 1);
  assert(g() == 3);

  auto h = std::bind_front(long_test, 1, 1, 1);
  assert(h(2, 2, 2) == 9);

  // Make sure the arg is passed by value.
  auto i = std::bind_front(add, n, 1);
  n = 100;
  assert(i() == 3);

  CopyMoveInfo info;
  auto copied = std::bind_front(wasCopied, info);
  assert(copied());

  auto moved = std::bind_front(wasMoved, info);
  assert(std::move(moved)());
}

struct variadic_fn {
  template <class... Args>
  constexpr int operator()(Args&&... args) {
    return sizeof...(args);
  }
};

constexpr void test_variadic() {
  variadic_fn value;
  auto fn = std::bind_front(value, 0, 0, 0);
  assert(fn(0, 0, 0) == 6);
}

struct mutable_callable {
  bool should_call_const;

  constexpr bool operator()(int, int) {
    assert(!should_call_const);
    return true;
  }
  constexpr bool operator()(int, int) const {
    assert(should_call_const);
    return true;
  }
};

constexpr void test_mutable() {
  const mutable_callable v1{true};
  const auto fn1 = std::bind_front(v1, 0);
  assert(fn1(0));

  mutable_callable v2{false};
  auto fn2 = std::bind_front(v2, 0);
  assert(fn2(0));
};

struct call_member {
  constexpr bool member(int, int) { return true; }
};

constexpr void test_call_member() {
  call_member value;
  auto fn = std::bind_front(&call_member::member, value, 0);
  assert(fn(0));
}

struct no_const_lvalue {
  constexpr void operator()(int) && {};
};

constexpr auto make_no_const_lvalue(int x) {
  // This is to test that bind_front works when something like the following would not:
  // return [nc = no_const_lvalue{}, x] { return nc(x); };
  // Above would not work because it would look for a () const & overload.
  return std::bind_front(no_const_lvalue{}, x);
}

constexpr void test_no_const_lvalue() { make_no_const_lvalue(1)(); }

constexpr void constructor_tests() {
  {
    MoveOnlyCallable value(true);
    using RetT = decltype(std::bind_front(std::move(value), 1));

    static_assert(std::is_move_constructible<RetT>::value);
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

    static_assert(std::is_move_constructible<RetT>::value);
    static_assert(std::is_copy_constructible<RetT>::value);
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

    static_assert(std::is_move_constructible<RetT>::value);
    static_assert(!std::is_copy_constructible<RetT>::value);
    static_assert(std::is_move_assignable<RetT>::value);
    static_assert(!std::is_copy_assignable<RetT>::value);
  }
}

template <class Res, class F, class... Args>
constexpr void test_return(F&& value, Args&&... args) {
  auto ret =
      std::bind_front(std::forward<F>(value), std::forward<Args>(args)...);
  static_assert(std::is_same<decltype(ret()), Res>::value);
}

constexpr void test_return_types() {
  test_return<Foo>(FooCall{}, 1, 2);
  test_return<bool>(S{}, 1);
  test_return<int>(add, 2, 2);
}

constexpr void test_arg_count() {
  using T = decltype(std::bind_front(add, 1));
  static_assert(!std::is_invocable<T>::value);
  static_assert(std::is_invocable<T, int>::value);
}

template <class... Args>
struct is_bind_frontable {
  template <class... LocalArgs>
  static auto test(int)
      -> decltype((void)std::bind_front(std::declval<LocalArgs>()...),
                  std::true_type());

  template <class...>
  static std::false_type test(...);

  static constexpr bool value = decltype(test<Args...>(0))::value;
};

struct NotCopyMove {
  NotCopyMove() = delete;
  NotCopyMove(const NotCopyMove&) = delete;
  NotCopyMove(NotCopyMove&&) = delete;
  void operator()() {}
};

struct NonConstCopyConstructible {
  explicit NonConstCopyConstructible() {}
  NonConstCopyConstructible(NonConstCopyConstructible&) {}
};

struct MoveConstructible {
  explicit MoveConstructible() {}
  MoveConstructible(MoveConstructible&&) {}
};

constexpr void test_invocability() {
  static_assert(!std::is_constructible_v<NotCopyMove, NotCopyMove>);
  static_assert(!std::is_move_constructible_v<NotCopyMove>);
  static_assert(!is_bind_frontable<NotCopyMove>::value);
  static_assert(!is_bind_frontable<NotCopyMove&>::value);

  static_assert(
      !std::is_constructible_v<MoveConstructible, MoveConstructible&>);
  static_assert(std::is_move_constructible_v<MoveConstructible>);
  static_assert(is_bind_frontable<variadic_fn, MoveConstructible>::value);
  static_assert(
      !is_bind_frontable<variadic_fn, MoveConstructible&>::value);

  static_assert(std::is_constructible_v<NonConstCopyConstructible,
                                        NonConstCopyConstructible&>);
  static_assert(!std::is_move_constructible_v<NonConstCopyConstructible>);
  static_assert(
      !is_bind_frontable<variadic_fn, NonConstCopyConstructible&>::value);
  static_assert(
      !is_bind_frontable<variadic_fn, NonConstCopyConstructible>::value);
}

constexpr bool test() {
  basic_tests();
  constructor_tests();
  test_return_types();
  test_arg_count();
  test_variadic();
  test_mutable();
  test_call_member();
  test_no_const_lvalue();
  test_invocability();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
