//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template<class... Types>
//     tuple<Types&&...> forward_as_tuple(Types&&... t);

#include <tuple>
#include <cassert>

template <class Tuple>
void
test0(const Tuple& t)
{
    static_assert(std::tuple_size<Tuple>::value == 0, "");
}

template <class Tuple>
void
test1a(const Tuple& t)
{
    static_assert(std::tuple_size<Tuple>::value == 1, "");
    static_assert(std::is_same<typename std::tuple_element<0, Tuple>::type, int&&>::value, "");
    assert(std::get<0>(t) == 1);
}

template <class Tuple>
void
test1b(const Tuple& t)
{
    static_assert(std::tuple_size<Tuple>::value == 1, "");
    static_assert(std::is_same<typename std::tuple_element<0, Tuple>::type, int&>::value, "");
    assert(std::get<0>(t) == 2);
}

template <class Tuple>
void
test2a(const Tuple& t)
{
    static_assert(std::tuple_size<Tuple>::value == 2, "");
    static_assert(std::is_same<typename std::tuple_element<0, Tuple>::type, double&>::value, "");
    static_assert(std::is_same<typename std::tuple_element<1, Tuple>::type, char&>::value, "");
    assert(std::get<0>(t) == 2.5);
    assert(std::get<1>(t) == 'a');
}

int main()
{
    {
        test0(std::forward_as_tuple());
    }
    {
        test1a(std::forward_as_tuple(1));
    }
    {
        int i = 2;
        test1b(std::forward_as_tuple(i));
    }
    {
        double i = 2.5;
        char c = 'a';
        test2a(std::forward_as_tuple(i, c));
    }
}
