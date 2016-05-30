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

// constexpr const T* optional<T>::operator->() const;

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
};

struct Y
{
    int test() const {return 2;}
};

struct Z
{
    const Z* operator&() const {return this;}
    constexpr int test() const {return 1;}
};

int main()
{
    {
        constexpr optional<X> opt(X{});
        static_assert(opt->test() == 3, "");
    }
    {
        constexpr optional<Y> opt(Y{});
        assert(opt->test() == 2);
    }
    {
        constexpr optional<Z> opt(Z{});
        assert(opt->test() == 1);
    }
#ifdef _LIBCPP_DEBUG
    {
        const optional<X> opt;
        assert(opt->test() == 3);
        assert(false);
    }
#endif  // _LIBCPP_DEBUG
}
