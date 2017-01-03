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
#include <type_traits>

template <class T, class = decltype(std::tuple_size<T>::value)>
constexpr bool has_value(int) { return true; }
template <class> constexpr bool has_value(long) { return false; }
template <class T> constexpr bool has_value() { return has_value<T>(0); }

struct Dummy {};

template <class T, std::size_t N>
void test()
{
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<const T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<volatile T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<const volatile T> >::value), "");
}

void test_tuple_size_value_sfinae() {
  // Test that the ::value member does not exist
  static_assert(has_value<std::tuple<int> const>(), "");
  static_assert(has_value<std::pair<int, long> volatile>(), "");
  static_assert(!has_value<int>(), "");
  static_assert(!has_value<const int>(), "");
  static_assert(!has_value<volatile void>(), "");
  static_assert(!has_value<const volatile std::tuple<int>&>(), "");
}

int main()
{
    test<std::tuple<>, 0>();
    test<std::tuple<int>, 1>();
    test<std::tuple<char, int>, 2>();
    test<std::tuple<char, char*, int>, 3>();
    test_tuple_size_value_sfinae();
}
