//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template<Returnable R, CopyConstructible... ArgTypes>
// class function<R(ArgTypes...)>
//   : public unary_function<T1, R>      // iff sizeof...(ArgTypes) == 1 and
//                                       // ArgTypes contains T1
//   : public binary_function<T1, T2, R> // iff sizeof...(ArgTypes) == 2 and
//                                       // ArgTypes contains T1 and T2
// {
// public:
//     typedef R result_type;
//     ...
// };

#include <functional>
#include <type_traits>

int main()
{
    static_assert((!std::is_base_of<std::unary_function <int, int>,
                                           std::function<int()> >::value), "");
    static_assert((!std::is_base_of<std::binary_function<int, int, int>,
                                           std::function<int()> >::value), "");
    static_assert(( std::is_same<          std::function<int()>::result_type,
                                                         int>::value), "");

    static_assert(( std::is_base_of<std::unary_function <int, double>,
                                           std::function<double(int)> >::value), "");
    static_assert((!std::is_base_of<std::binary_function<int, int, double>,
                                           std::function<double(int)> >::value), "");
    static_assert(( std::is_same<          std::function<double(int)>::result_type,
                                                         double>::value), "");

    static_assert((!std::is_base_of<std::unary_function <int, double>,
                                           std::function<double(int, char)> >::value), "");
    static_assert(( std::is_base_of<std::binary_function<int, char, double>,
                                           std::function<double(int, char)> >::value), "");
    static_assert(( std::is_same<          std::function<double(int, char)>::result_type,
                                                         double>::value), "");
}
