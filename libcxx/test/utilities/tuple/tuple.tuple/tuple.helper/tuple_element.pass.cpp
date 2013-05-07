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

// template <size_t I, class... Types>
// class tuple_element<I, tuple<Types...> >
// {
// public:
//     typedef Ti type;
// };

#include <tuple>
#include <type_traits>

template <class T, std::size_t N, class U>
void test()
{
    static_assert((std::is_same<typename std::tuple_element<N, T>::type, U>::value), "");
    static_assert((std::is_same<typename std::tuple_element<N, const T>::type, const U>::value), "");
    static_assert((std::is_same<typename std::tuple_element<N, volatile T>::type, volatile U>::value), "");
    static_assert((std::is_same<typename std::tuple_element<N, const volatile T>::type, const volatile U>::value), "");
}
int main()
{
    test<std::tuple<int>, 0, int>();
    test<std::tuple<char, int>, 0, char>();
    test<std::tuple<char, int>, 1, int>();
    test<std::tuple<int*, char, int>, 0, int*>();
    test<std::tuple<int*, char, int>, 1, char>();
    test<std::tuple<int*, char, int>, 2, int>();
}
