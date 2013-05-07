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

// template <class... UTypes> tuple(const tuple<UTypes...>& u);

#include <tuple>
#include <string>
#include <cassert>

struct B
{
    int id_;

    explicit B(int i) : id_(i) {}
};

struct D
    : B
{
    explicit D(int i) : B(i) {}
};

int main()
{
    {
        typedef std::tuple<double> T0;
        typedef std::tuple<int> T1;
        T0 t0(2.5);
        T1 t1 = t0;
        assert(std::get<0>(t1) == 2);
    }
    {
        typedef std::tuple<double, char> T0;
        typedef std::tuple<int, int> T1;
        T0 t0(2.5, 'a');
        T1 t1 = t0;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
    }
    {
        typedef std::tuple<double, char, D> T0;
        typedef std::tuple<int, int, B> T1;
        T0 t0(2.5, 'a', D(3));
        T1 t1 = t0;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 3);
    }
    {
        D d(3);
        typedef std::tuple<double, char, D&> T0;
        typedef std::tuple<int, int, B&> T1;
        T0 t0(2.5, 'a', d);
        T1 t1 = t0;
        d.id_ = 2;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 2);
    }
    {
        typedef std::tuple<double, char, int> T0;
        typedef std::tuple<int, int, B> T1;
        T0 t0(2.5, 'a', 3);
        T1 t1(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 3);
    }
}
