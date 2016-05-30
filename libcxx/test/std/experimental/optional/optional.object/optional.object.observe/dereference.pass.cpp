//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <optional>

// T& optional<T>::operator*();

#ifdef _LIBCPP_DEBUG
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))
#endif

#include <experimental/optional>
#include <type_traits>
#include <cassert>

using std::experimental::optional;

struct X
{
    constexpr int test() const {return 3;}
    int test() {return 4;}
};

int main()
{
    {
        optional<X> opt(X{});
        assert((*opt).test() == 4);
    }
#ifdef _LIBCPP_DEBUG
    {
        optional<X> opt;
        assert((*opt).test() == 3);
        assert(false);
    }
#endif  // _LIBCPP_DEBUG
}
