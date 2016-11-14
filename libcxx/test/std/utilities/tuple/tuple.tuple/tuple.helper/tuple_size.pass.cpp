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

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <utility>
#include <array>
#include <type_traits>

template <class T, class = decltype(std::tuple_size<T>::value)>
constexpr bool has_value(int) { return true; }
template <class> constexpr bool has_value(long) { return false; }
template <class T> constexpr bool has_value() { return has_value<T>(0); }


template <class T, std::size_t N>
void test()
{
    static_assert(has_value<T>(), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<T> >::value), "");
    static_assert(has_value<const T>(), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<const T> >::value), "");
    static_assert(has_value<volatile T>(), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<volatile T> >::value), "");

    static_assert(has_value<const volatile T>(), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<const volatile T> >::value), "");
    {
        static_assert(!has_value<T &>(), "");
        static_assert(!has_value<T *>(), "");
    }
}

int main()
{
    test<std::tuple<>, 0>();
    test<std::tuple<int>, 1>();
    test<std::tuple<char, int>, 2>();
    test<std::tuple<char, char*, int>, 3>();
    test<std::pair<int, void*>, 2>();
    test<std::array<int, 42>, 42>();
    {
        static_assert(!has_value<void>(), "");
        static_assert(!has_value<void*>(), "");
        static_assert(!has_value<int>(), "");
        static_assert(!has_value<std::pair<int, int>*>(), "");
        static_assert(!has_value<std::array<int, 42>&>(), "");
    }
}
