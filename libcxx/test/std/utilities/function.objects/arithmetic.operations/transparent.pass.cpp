//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
#include <functional>
#include <string>

template <class T>
struct is_transparent
{
private:
    struct two {char lx; char lxx;};
    template <class U> static two test(...);
    template <class U> static char test(typename U::is_transparent* = 0);
public:
    static const bool value = sizeof(test<T>(0)) == 1;
};


int main(int, char**)
{
    static_assert ( !is_transparent<std::plus<int>>::value, "" );
    static_assert ( !is_transparent<std::plus<std::string>>::value, "" );
    static_assert (  is_transparent<std::plus<void>>::value, "" );
    static_assert (  is_transparent<std::plus<>>::value, "" );

    static_assert ( !is_transparent<std::minus<int>>::value, "" );
    static_assert ( !is_transparent<std::minus<std::string>>::value, "" );
    static_assert (  is_transparent<std::minus<void>>::value, "" );
    static_assert (  is_transparent<std::minus<>>::value, "" );

    static_assert ( !is_transparent<std::multiplies<int>>::value, "" );
    static_assert ( !is_transparent<std::multiplies<std::string>>::value, "" );
    static_assert (  is_transparent<std::multiplies<void>>::value, "" );
    static_assert (  is_transparent<std::multiplies<>>::value, "" );

    static_assert ( !is_transparent<std::divides<int>>::value, "" );
    static_assert ( !is_transparent<std::divides<std::string>>::value, "" );
    static_assert (  is_transparent<std::divides<void>>::value, "" );
    static_assert (  is_transparent<std::divides<>>::value, "" );

    static_assert ( !is_transparent<std::modulus<int>>::value, "" );
    static_assert ( !is_transparent<std::modulus<std::string>>::value, "" );
    static_assert (  is_transparent<std::modulus<void>>::value, "" );
    static_assert (  is_transparent<std::modulus<>>::value, "" );

    static_assert ( !is_transparent<std::negate<int>>::value, "" );
    static_assert ( !is_transparent<std::negate<std::string>>::value, "" );
    static_assert (  is_transparent<std::negate<void>>::value, "" );
    static_assert (  is_transparent<std::negate<>>::value, "" );

    return 0;
}
