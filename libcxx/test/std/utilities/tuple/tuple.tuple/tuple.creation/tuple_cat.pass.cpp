//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Tuples> tuple<CTypes...> tuple_cat(Tuples&&... tpls);

// UNSUPPORTED: c++03

#include <tuple>
#include <utility>
#include <array>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

namespace NS {
struct Namespaced {
  int i;
};
template<typename ...Ts>
void forward_as_tuple(Ts...) = delete;
}

// https://llvm.org/PR41689
struct Unconstrained {
  int data;
  template <typename Arg>
  TEST_CONSTEXPR_CXX14 Unconstrained(Arg arg) : data(arg) {}
};

TEST_CONSTEXPR_CXX14 bool test_tuple_cat_with_unconstrained_constructor() {
  {
    auto tup_src = std::tuple<Unconstrained>(Unconstrained(5));
    auto tup     = std::tuple_cat(tup_src);
    assert(std::get<0>(tup).data == 5);
  }
  {
    auto tup = std::tuple_cat(std::tuple<Unconstrained>(Unconstrained(6)));
    assert(std::get<0>(tup).data == 6);
  }
  {
    auto tup = std::tuple_cat(std::tuple<Unconstrained>(Unconstrained(7)), std::tuple<>());
    assert(std::get<0>(tup).data == 7);
  }
#if TEST_STD_VER >= 17
  {
    auto tup_src = std::tuple(Unconstrained(8));
    auto tup     = std::tuple_cat(tup_src);
    ASSERT_SAME_TYPE(decltype(tup), std::tuple<Unconstrained>);
    assert(std::get<0>(tup).data == 8);
  }
  {
    auto tup = std::tuple_cat(std::tuple(Unconstrained(9)));
    ASSERT_SAME_TYPE(decltype(tup), std::tuple<Unconstrained>);
    assert(std::get<0>(tup).data == 9);
  }
  {
    auto tup = std::tuple_cat(std::tuple(Unconstrained(10)), std::tuple());
    ASSERT_SAME_TYPE(decltype(tup), std::tuple<Unconstrained>);
    assert(std::get<0>(tup).data == 10);
  }
#endif
  return true;
}

int main(int, char**)
{
    {
        std::tuple<> t = std::tuple_cat();
        ((void)t); // Prevent unused warning
    }
    {
        std::tuple<> t1;
        std::tuple<> t2 = std::tuple_cat(t1);
        ((void)t2); // Prevent unused warning
    }
    {
        std::tuple<> t = std::tuple_cat(std::tuple<>());
        ((void)t); // Prevent unused warning
    }
    {
        std::tuple<> t = std::tuple_cat(std::array<int, 0>());
        ((void)t); // Prevent unused warning
    }
    {
        std::tuple<int> t1(1);
        std::tuple<int> t = std::tuple_cat(t1);
        assert(std::get<0>(t) == 1);
    }

#if TEST_STD_VER > 11
    {
        constexpr std::tuple<> t = std::tuple_cat();
        ((void)t); // Prevent unused warning
    }
    {
        constexpr std::tuple<> t1;
        constexpr std::tuple<> t2 = std::tuple_cat(t1);
        ((void)t2); // Prevent unused warning
    }
    {
        constexpr std::tuple<> t = std::tuple_cat(std::tuple<>());
        ((void)t); // Prevent unused warning
    }
    {
        constexpr std::tuple<> t = std::tuple_cat(std::array<int, 0>());
        ((void)t); // Prevent unused warning
    }
    {
        constexpr std::tuple<int> t1(1);
        constexpr std::tuple<int> t = std::tuple_cat(t1);
        static_assert(std::get<0>(t) == 1, "");
    }
    {
        constexpr std::tuple<int> t1(1);
        constexpr std::tuple<int, int> t = std::tuple_cat(t1, t1);
        static_assert(std::get<0>(t) == 1, "");
        static_assert(std::get<1>(t) == 1, "");
    }
#endif
    {
        std::tuple<int, MoveOnly> t =
                                std::tuple_cat(std::tuple<int, MoveOnly>(1, 2));
        assert(std::get<0>(t) == 1);
        assert(std::get<1>(t) == 2);
    }
    {
        std::tuple<int, int, int> t = std::tuple_cat(std::array<int, 3>());
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 0);
        assert(std::get<2>(t) == 0);
    }
    {
        std::tuple<int, MoveOnly> t = std::tuple_cat(std::pair<int, MoveOnly>(2, 1));
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == 1);
    }

    {
        std::tuple<> t1;
        std::tuple<> t2;
        std::tuple<> t3 = std::tuple_cat(t1, t2);
        ((void)t3); // Prevent unused warning
    }
    {
        std::tuple<> t1;
        std::tuple<int> t2(2);
        std::tuple<int> t3 = std::tuple_cat(t1, t2);
        assert(std::get<0>(t3) == 2);
    }
    {
        std::tuple<> t1;
        std::tuple<int> t2(2);
        std::tuple<int> t3 = std::tuple_cat(t2, t1);
        assert(std::get<0>(t3) == 2);
    }
    {
        std::tuple<int*> t1;
        std::tuple<int> t2(2);
        std::tuple<int*, int> t3 = std::tuple_cat(t1, t2);
        assert(std::get<0>(t3) == nullptr);
        assert(std::get<1>(t3) == 2);
    }
    {
        std::tuple<int*> t1;
        std::tuple<int> t2(2);
        std::tuple<int, int*> t3 = std::tuple_cat(t2, t1);
        assert(std::get<0>(t3) == 2);
        assert(std::get<1>(t3) == nullptr);
    }
    {
        std::tuple<int*> t1;
        std::tuple<int, double> t2(2, 3.5);
        std::tuple<int*, int, double> t3 = std::tuple_cat(t1, t2);
        assert(std::get<0>(t3) == nullptr);
        assert(std::get<1>(t3) == 2);
        assert(std::get<2>(t3) == 3.5);
    }
    {
        std::tuple<int*> t1;
        std::tuple<int, double> t2(2, 3.5);
        std::tuple<int, double, int*> t3 = std::tuple_cat(t2, t1);
        assert(std::get<0>(t3) == 2);
        assert(std::get<1>(t3) == 3.5);
        assert(std::get<2>(t3) == nullptr);
    }
    {
        std::tuple<int*, MoveOnly> t1(nullptr, 1);
        std::tuple<int, double> t2(2, 3.5);
        std::tuple<int*, MoveOnly, int, double> t3 =
                                              std::tuple_cat(std::move(t1), t2);
        assert(std::get<0>(t3) == nullptr);
        assert(std::get<1>(t3) == 1);
        assert(std::get<2>(t3) == 2);
        assert(std::get<3>(t3) == 3.5);
    }
    {
        std::tuple<int*, MoveOnly> t1(nullptr, 1);
        std::tuple<int, double> t2(2, 3.5);
        std::tuple<int, double, int*, MoveOnly> t3 =
                                              std::tuple_cat(t2, std::move(t1));
        assert(std::get<0>(t3) == 2);
        assert(std::get<1>(t3) == 3.5);
        assert(std::get<2>(t3) == nullptr);
        assert(std::get<3>(t3) == 1);
    }
    {
        std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        std::tuple<int*, MoveOnly> t2(nullptr, 4);
        std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   std::tuple_cat(std::move(t1), std::move(t2));
        assert(std::get<0>(t3) == 1);
        assert(std::get<1>(t3) == 2);
        assert(std::get<2>(t3) == nullptr);
        assert(std::get<3>(t3) == 4);
    }

    {
        std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        std::tuple<int*, MoveOnly> t2(nullptr, 4);
        std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   std::tuple_cat(std::tuple<>(),
                                                  std::move(t1),
                                                  std::move(t2));
        assert(std::get<0>(t3) == 1);
        assert(std::get<1>(t3) == 2);
        assert(std::get<2>(t3) == nullptr);
        assert(std::get<3>(t3) == 4);
    }
    {
        std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        std::tuple<int*, MoveOnly> t2(nullptr, 4);
        std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   std::tuple_cat(std::move(t1),
                                                  std::tuple<>(),
                                                  std::move(t2));
        assert(std::get<0>(t3) == 1);
        assert(std::get<1>(t3) == 2);
        assert(std::get<2>(t3) == nullptr);
        assert(std::get<3>(t3) == 4);
    }
    {
        std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        std::tuple<int*, MoveOnly> t2(nullptr, 4);
        std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   std::tuple_cat(std::move(t1),
                                                  std::move(t2),
                                                  std::tuple<>());
        assert(std::get<0>(t3) == 1);
        assert(std::get<1>(t3) == 2);
        assert(std::get<2>(t3) == nullptr);
        assert(std::get<3>(t3) == 4);
    }
    {
        std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        std::tuple<int*, MoveOnly> t2(nullptr, 4);
        std::tuple<MoveOnly, MoveOnly, int*, MoveOnly, int> t3 =
                                   std::tuple_cat(std::move(t1),
                                                  std::move(t2),
                                                  std::tuple<int>(5));
        assert(std::get<0>(t3) == 1);
        assert(std::get<1>(t3) == 2);
        assert(std::get<2>(t3) == nullptr);
        assert(std::get<3>(t3) == 4);
        assert(std::get<4>(t3) == 5);
    }
    {
        // See bug #19616.
        auto t1 = std::tuple_cat(
            std::make_tuple(std::make_tuple(1)),
            std::make_tuple()
        );
        assert(t1 == std::make_tuple(std::make_tuple(1)));

        auto t2 = std::tuple_cat(
            std::make_tuple(std::make_tuple(1)),
            std::make_tuple(std::make_tuple(2))
        );
        assert(t2 == std::make_tuple(std::make_tuple(1), std::make_tuple(2)));
    }
    {
        int x = 101;
        std::tuple<int, const int, int&, const int&, int&&> t(42, 101, x, x, std::move(x));
        const auto& ct = t;
        std::tuple<int, const int, int&, const int&> t2(42, 101, x, x);
        const auto& ct2 = t2;

        auto r = std::tuple_cat(std::move(t), std::move(ct), t2, ct2);

        ASSERT_SAME_TYPE(decltype(r), std::tuple<
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&,
            int, const int, int&, const int&>);
        ((void)r);
    }
    {
        std::tuple<NS::Namespaced> t1(NS::Namespaced{1});
        std::tuple<NS::Namespaced> t = std::tuple_cat(t1);
        std::tuple<NS::Namespaced, NS::Namespaced> t2 =
            std::tuple_cat(t1, t1);
        assert(std::get<0>(t).i == 1);
        assert(std::get<0>(t2).i == 1);
    }
    // See https://llvm.org/PR41689
    {
      test_tuple_cat_with_unconstrained_constructor();
#if TEST_STD_VER >= 14
      static_assert(test_tuple_cat_with_unconstrained_constructor(), "");
#endif
    }

    return 0;
}
