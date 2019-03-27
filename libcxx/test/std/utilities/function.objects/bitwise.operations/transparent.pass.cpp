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


int main(int, char**) {
    static_assert ( !is_transparent<std::bit_and<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_and<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_and<void>>::value, "" );
    static_assert (  is_transparent<std::bit_and<>>::value, "" );

    static_assert ( !is_transparent<std::bit_or<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_or<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_or<void>>::value, "" );
    static_assert (  is_transparent<std::bit_or<>>::value, "" );

    static_assert ( !is_transparent<std::bit_xor<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_xor<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_xor<void>>::value, "" );
    static_assert (  is_transparent<std::bit_xor<>>::value, "" );

    static_assert ( !is_transparent<std::bit_not<int>>::value, "" );
    static_assert ( !is_transparent<std::bit_not<std::string>>::value, "" );
    static_assert (  is_transparent<std::bit_not<void>>::value, "" );
    static_assert (  is_transparent<std::bit_not<>>::value, "" );

    return 0;
}
