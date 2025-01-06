//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <tuple>

// template <class T, class Tuple> constexpr T make_from_tuple(Tuple&&);

#include <tuple>
#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

#include "test_macros.h"
#include "type_id.h"

template <class Tuple>
struct ConstexprConstructibleFromTuple {
  template <class ...Args>
  explicit constexpr ConstexprConstructibleFromTuple(Args&&... xargs)
      : args{std::forward<Args>(xargs)...} {}
  Tuple args;
};

template <class TupleLike>
struct ConstructibleFromTuple;

template <template <class ...> class Tuple, class ...Types>
struct ConstructibleFromTuple<Tuple<Types...>> {
  template <class ...Args>
  explicit ConstructibleFromTuple(Args&&... xargs)
      : args(xargs...),
        arg_types(&makeArgumentID<Args&&...>())
  {}
  Tuple<std::decay_t<Types>...> args;
  TypeID const* arg_types;
};

template <class Tp, std::size_t N>
struct ConstructibleFromTuple<std::array<Tp, N>> {
template <class ...Args>
  explicit ConstructibleFromTuple(Args&&... xargs)
      : args{xargs...},
        arg_types(&makeArgumentID<Args&&...>())
  {}
  std::array<Tp, N> args;
  TypeID const* arg_types;
};

template <class Tuple>
constexpr bool do_constexpr_test(Tuple&& tup) {
    using RawTuple = std::decay_t<Tuple>;
    using Tp = ConstexprConstructibleFromTuple<RawTuple>;
    return std::make_from_tuple<Tp>(std::forward<Tuple>(tup)).args == tup;
}

template <class ...ExpectTypes, class Tuple>
bool do_forwarding_test(Tuple&& tup) {
    using RawTuple = std::decay_t<Tuple>;
    using Tp = ConstructibleFromTuple<RawTuple>;
    const Tp value = std::make_from_tuple<Tp>(std::forward<Tuple>(tup));
    return value.args == tup
        && value.arg_types == &makeArgumentID<ExpectTypes...>();
}

void test_constexpr_construction() {
    {
        constexpr std::tuple<> tup;
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr std::tuple<int> tup(42);
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr std::tuple<int, long, void*> tup(42, 101, nullptr);
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr std::pair<int, const char*> p(42, "hello world");
        static_assert(do_constexpr_test(p), "");
    }
    {
        using Tuple = std::array<int, 3>;
        using ValueTp = ConstexprConstructibleFromTuple<Tuple>;
        constexpr Tuple arr = {42, 101, -1};
        constexpr ValueTp value = std::make_from_tuple<ValueTp>(arr);
        static_assert(value.args[0] == arr[0] && value.args[1] == arr[1]
            && value.args[2] == arr[2], "");
    }
}

void test_perfect_forwarding() {
    {
        using Tup = std::tuple<>;
        Tup tup;
        Tup const& ctup = tup;
        assert(do_forwarding_test<>(tup));
        assert(do_forwarding_test<>(ctup));
    }
    {
        using Tup = std::tuple<int>;
        Tup tup(42);
        Tup const& ctup = tup;
        assert(do_forwarding_test<int&>(tup));
        assert(do_forwarding_test<int const&>(ctup));
        assert(do_forwarding_test<int&&>(std::move(tup)));
        assert(do_forwarding_test<int const&&>(std::move(ctup)));
    }
    {
        using Tup = std::tuple<int&, const char*, unsigned&&>;
        int x = 42;
        unsigned y = 101;
        Tup tup(x, "hello world", std::move(y));
        Tup const& ctup = tup;
        assert((do_forwarding_test<int&, const char*&, unsigned&>(tup)));
        assert((do_forwarding_test<int&, const char* const&, unsigned &>(ctup)));
        assert((do_forwarding_test<int&, const char*&&, unsigned&&>(std::move(tup))));
        assert((do_forwarding_test<int&, const char* const&&, unsigned &&>(std::move(ctup))));
    }
    // test with pair<T, U>
    {
        using Tup = std::pair<int&, const char*>;
        int x = 42;
        Tup tup(x, "hello world");
        Tup const& ctup = tup;
        assert((do_forwarding_test<int&, const char*&>(tup)));
        assert((do_forwarding_test<int&, const char* const&>(ctup)));
        assert((do_forwarding_test<int&, const char*&&>(std::move(tup))));
        assert((do_forwarding_test<int&, const char* const&&>(std::move(ctup))));
    }
    // test with array<T, I>
    {
        using Tup = std::array<int, 3>;
        Tup tup = {42, 101, -1};
        Tup const& ctup = tup;
        assert((do_forwarding_test<int&, int&, int&>(tup)));
        assert((do_forwarding_test<int const&, int const&, int const&>(ctup)));
        assert((do_forwarding_test<int&&, int&&, int&&>(std::move(tup))));
        assert((do_forwarding_test<int const&&, int const&&, int const&&>(std::move(ctup))));
    }
}

void test_noexcept() {
    struct NothrowMoveable {
      NothrowMoveable() = default;
      NothrowMoveable(NothrowMoveable const&) {}
      NothrowMoveable(NothrowMoveable&&) noexcept {}
    };
    struct TestType {
      TestType(int, NothrowMoveable) noexcept {}
      TestType(int, int, int) noexcept(false) {}
      TestType(long, long, long) noexcept {}
    };
    {
        using Tuple = std::tuple<int, NothrowMoveable>;
        Tuple tup; ((void)tup);
        Tuple const& ctup = tup; ((void)ctup);
        ASSERT_NOT_NOEXCEPT(std::make_from_tuple<TestType>(ctup));
        LIBCPP_ASSERT_NOEXCEPT(std::make_from_tuple<TestType>(std::move(tup)));
    }
    {
        using Tuple = std::pair<int, NothrowMoveable>;
        Tuple tup; ((void)tup);
        Tuple const& ctup = tup; ((void)ctup);
        ASSERT_NOT_NOEXCEPT(std::make_from_tuple<TestType>(ctup));
        LIBCPP_ASSERT_NOEXCEPT(std::make_from_tuple<TestType>(std::move(tup)));
    }
    {
        using Tuple = std::tuple<int, int, int>;
        Tuple tup; ((void)tup);
        ASSERT_NOT_NOEXCEPT(std::make_from_tuple<TestType>(tup));
    }
    {
        using Tuple = std::tuple<long, long, long>;
        Tuple tup; ((void)tup);
        LIBCPP_ASSERT_NOEXCEPT(std::make_from_tuple<TestType>(tup));
    }
    {
        using Tuple = std::array<int, 3>;
        Tuple tup; ((void)tup);
        ASSERT_NOT_NOEXCEPT(std::make_from_tuple<TestType>(tup));
    }
    {
        using Tuple = std::array<long, 3>;
        Tuple tup; ((void)tup);
        LIBCPP_ASSERT_NOEXCEPT(std::make_from_tuple<TestType>(tup));
    }
}

namespace LWG3528 {
template <class T, class Tuple>
auto test_make_from_tuple(T&&, Tuple&& t) -> decltype(std::make_from_tuple<T>(t), std::uint8_t()) {
  return 0;
}
template <class T, class Tuple>
uint32_t test_make_from_tuple(...) {
  return 0;
}

template <class T, class Tuple>
static constexpr bool can_make_from_tuple =
    std::is_same_v<decltype(test_make_from_tuple<T, Tuple>(T{}, Tuple{})), std::uint8_t>;

#ifdef _LIBCPP_VERSION
template <class T, class Tuple>
auto test_make_from_tuple_impl(T&&, Tuple&& t)
    -> decltype(std::__make_from_tuple_impl<T>(
                    t, typename std::__make_tuple_indices< std::tuple_size_v<std::remove_reference_t<Tuple>>>::type{}),
                std::uint8_t()) {
  return 0;
}
template <class T, class Tuple>
uint32_t test_make_from_tuple_impl(...) {
  return 0;
}

template <class T, class Tuple>
static constexpr bool can_make_from_tuple_impl =
    std::is_same_v<decltype(test_make_from_tuple_impl<T, Tuple>(T{}, Tuple{})), std::uint8_t>;
#endif // _LIBCPP_VERSION

struct A {
  int a;
};
struct B : public A {};

struct C {
  C(const B&) {}
};

enum class D {
  ONE,
  TWO,
};

// Test std::make_from_tuple constraints.

// reinterpret_cast
static_assert(!can_make_from_tuple<int*, std::tuple<A*>>);
static_assert(can_make_from_tuple<A*, std::tuple<A*>>);

// const_cast
static_assert(!can_make_from_tuple<char*, std::tuple<const char*>>);
static_assert(!can_make_from_tuple<volatile char*, std::tuple<const volatile char*>>);
static_assert(can_make_from_tuple<volatile char*, std::tuple<volatile char*>>);
static_assert(can_make_from_tuple<char*, std::tuple<char*>>);
static_assert(can_make_from_tuple<const char*, std::tuple<char*>>);
static_assert(can_make_from_tuple<const volatile char*, std::tuple<volatile char*>>);

// static_cast
static_assert(!can_make_from_tuple<int, std::tuple<D>>);
static_assert(!can_make_from_tuple<D, std::tuple<int>>);
static_assert(can_make_from_tuple<long, std::tuple<int>>);
static_assert(can_make_from_tuple<double, std::tuple<float>>);
static_assert(can_make_from_tuple<float, std::tuple<double>>);

// Test std::__make_from_tuple_impl constraints.

// reinterpret_cast
LIBCPP_STATIC_ASSERT(!can_make_from_tuple_impl<int*, std::tuple<A*>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<A*, std::tuple<A*>>);

// const_cast
LIBCPP_STATIC_ASSERT(!can_make_from_tuple_impl<char*, std::tuple<const char*>>);
LIBCPP_STATIC_ASSERT(!can_make_from_tuple_impl<volatile char*, std::tuple<const volatile char*>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<volatile char*, std::tuple<volatile char*>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<char*, std::tuple<char*>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<const char*, std::tuple<char*>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<const volatile char*, std::tuple<volatile char*>>);

// static_cast
LIBCPP_STATIC_ASSERT(!can_make_from_tuple_impl<int, std::tuple<D>>);
LIBCPP_STATIC_ASSERT(!can_make_from_tuple_impl<D, std::tuple<int>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<long, std::tuple<int>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<double, std::tuple<float>>);
LIBCPP_STATIC_ASSERT(can_make_from_tuple_impl<float, std::tuple<double>>);

} // namespace LWG3528

int main(int, char**)
{
    test_constexpr_construction();
    test_perfect_forwarding();
    test_noexcept();

  return 0;
}
