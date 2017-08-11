//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
#include <functional>
#include <string>

template <class T>
struct is_transparent
{
private:
    struct __two {char __lx; char __lxx;};
    template <class _Up> static __two __test(...);
    template <class _Up> static char __test(typename _Up::is_transparent* = 0);
public:
    static const bool value = sizeof(__test<T>(0)) == 1;
};


int main ()
{
    static_assert ( !is_transparent<std::logical_and<int>>::value, "" );
    static_assert ( !is_transparent<std::logical_and<std::string>>::value, "" );
    static_assert (  is_transparent<std::logical_and<void>>::value, "" );
    static_assert (  is_transparent<std::logical_and<>>::value, "" );

    static_assert ( !is_transparent<std::logical_or<int>>::value, "" );
    static_assert ( !is_transparent<std::logical_or<std::string>>::value, "" );
    static_assert (  is_transparent<std::logical_or<void>>::value, "" );
    static_assert (  is_transparent<std::logical_or<>>::value, "" );

    static_assert ( !is_transparent<std::logical_not<int>>::value, "" );
    static_assert ( !is_transparent<std::logical_not<std::string>>::value, "" );
    static_assert (  is_transparent<std::logical_not<void>>::value, "" );
    static_assert (  is_transparent<std::logical_not<>>::value, "" );

    return 0;
}
