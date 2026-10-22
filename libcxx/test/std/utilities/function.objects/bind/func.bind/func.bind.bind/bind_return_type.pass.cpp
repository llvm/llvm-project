//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// template<CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);
// template<Returnable R, CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);

// Check that the call operators have the proper return type and that they
// only SFINAE away when too few arguments are provided. Otherwise they should
// be well formed and should ignore any additional arguments.

#include <cassert>
#include <functional>
#include <type_traits>
#include <utility>

#include "test_macros.h"

int dummy = 42;

int return_value(int) { return dummy; }
int& return_lvalue(int) { return dummy; }
const int& return_const_lvalue(int) { return dummy; }
int&& return_rvalue(int) { return std::move(dummy); }
const int&& return_const_rvalue(int) { return std::move(dummy); }

template <class Bind, class ...Args>
auto CheckCallImp(int)
    -> decltype((std::declval<Bind>()(std::declval<Args>()...)), std::true_type{});

template <class Bind, class ...>
auto CheckCallImp(long) -> std::false_type;

template <class ...Args>
constexpr bool CheckCall() {
    return decltype(CheckCallImp<Args...>(0))::value;
}

template <class Expect, class Fn>
void do_test(Fn* func) {
    using namespace std::placeholders;
    auto ret = std::bind(func, _1);
    auto ret_r = std::bind<Expect>(func, _1);
    using Bind = decltype(ret);
    using BindR = decltype(ret_r);

    using Ret = decltype(ret(42));
    using Ret2 = decltype(ret(42, 43)); // Test that the extra argument is discarded.
    using RetR = decltype(ret_r(42));
    using RetR2 = decltype(ret_r(42, 43));

    static_assert(std::is_same<Ret, Expect>::value, "");
    static_assert(std::is_same<Ret2, Expect>::value, "");
    static_assert(std::is_same<RetR, Expect>::value, "");
    static_assert(std::is_same<RetR2, Expect>::value, "");

    Expect exp = ret(100); // the input value is ignored. dummy is returned.
    Expect exp2 = ret(101, 102);
    Expect exp_r = ret_r(100);
    Expect exp_r2 = ret_r(101, 102);

    assert(exp == 42);
    assert(exp2 == 42);
    assert(exp_r == 42);
    assert(exp_r2 == 42);

    if ((std::is_reference<Expect>::value)) {
        assert(&exp == &dummy);
        assert(&exp2 == &dummy);
        assert(&exp_r == &dummy);
        assert(&exp_r2 == &dummy);
    }
    // Check that the call operator SFINAE's away when too few arguments
    // are provided but is well-formed otherwise.
    {
        LIBCPP_STATIC_ASSERT(!CheckCall<Bind>(), "");
        static_assert(CheckCall<Bind, int>(), "");
        static_assert(CheckCall<Bind, int, int>(), "");
        LIBCPP_STATIC_ASSERT(!CheckCall<BindR>(), "");
        static_assert(CheckCall<BindR, int>(), "");
        static_assert(CheckCall<BindR, int, int>(), "");
    }
}


// Test but with an explicit return type which differs from the real one.
template <class Expect, class Fn>
void do_test_r(Fn* func) {
    using namespace std::placeholders;
    auto ret = std::bind<Expect>(func, _1);
    using Bind = decltype(ret);
    using Ret = decltype(ret(42));
    using Ret2 = decltype(ret(42, 43)); // Test that the extra argument is discarded.
    static_assert(std::is_same<Ret, Expect>::value, "");
    static_assert(std::is_same<Ret2, Expect>::value, "");
    Expect exp = ret(100); // the input value is ignored
    Expect exp2 = ret(101, 102);
    assert(exp == 42);
    assert(exp2 == 42);
    // Check that the call operator SFINAE's away when too few arguments
    // are provided but is well-formed otherwise.
    {
        LIBCPP_STATIC_ASSERT(!CheckCall<Bind>(), "");
        static_assert(CheckCall<Bind, int>(), "");
        static_assert(CheckCall<Bind, int, int>(), "");
    }
}

// P2255R2 disallows binding returned reference to temporary since C++23.
struct ConditionalTemporaryReturner {
  constexpr const int& operator()(std::false_type) const { return std::integral_constant<int, 42>::value; }
  constexpr int operator()(std::true_type) const { return 42; }
};

TEST_CONSTEXPR_CXX20 bool test_temporary_binding() {
#if TEST_STD_VER >= 23
  constexpr bool can_bind_to_temporary = false;
#else
  constexpr bool can_bind_to_temporary = true;
#endif

  auto bound1 = std::bind<const int&>(ConditionalTemporaryReturner{}, std::false_type{});

  static_assert(CheckCall<decltype(bound1)>(), "");
  static_assert(CheckCall<decltype(bound1)>(), "");
  static_assert(CheckCall<const decltype(bound1)>(), "");
  static_assert(CheckCall<const decltype(bound1)>(), "");

  assert(bound1() == 42);
  assert((&bound1() == &std::integral_constant<int, 42>::value));

  auto bound2 = std::bind<const int&>(ConditionalTemporaryReturner{}, std::true_type{});
  (void)bound2;

  static_assert(CheckCall<decltype(bound2)>() == can_bind_to_temporary, "");
  static_assert(CheckCall<decltype(bound2)>() == can_bind_to_temporary, "");
  static_assert(CheckCall<const decltype(bound2)>() == can_bind_to_temporary, "");
  static_assert(CheckCall<const decltype(bound2)>() == can_bind_to_temporary, "");

  return true;
}

int main(int, char**)
{
    do_test<int>(return_value);
    do_test<int&>(return_lvalue);
    do_test<const int&>(return_const_lvalue);
    do_test<int&&>(return_rvalue);
    do_test<const int&&>(return_const_rvalue);

    do_test_r<long>(return_value);
    do_test_r<long>(return_lvalue);
    do_test_r<long>(return_const_lvalue);
    do_test_r<long>(return_rvalue);
    do_test_r<long>(return_const_rvalue);

    test_temporary_binding();
#if TEST_STD_VER >= 20
    static_assert(test_temporary_binding());
#endif

    return 0;
}
