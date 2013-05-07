//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Tuples> tuple<CTypes...> tuple_cat(Tuples&&... tpls);

#include <tuple>
#include <utility>
#include <array>
#include <string>
#include <cassert>

#include "../MoveOnly.h"

int main()
{
    {
        std::tuple<> t = std::tuple_cat();
    }
    {
        std::tuple<> t1;
        std::tuple<> t2 = std::tuple_cat(t1);
    }
    {
        std::tuple<> t = std::tuple_cat(std::tuple<>());
    }
    {
        std::tuple<> t = std::tuple_cat(std::array<int, 0>());
    }

    {
        std::tuple<int> t1(1);
        std::tuple<int> t = std::tuple_cat(t1);
        assert(std::get<0>(t) == 1);
    }
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
}
