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

// template <class U> optional<T>& operator=(U&& v);

#include <experimental/optional>
#include <type_traits>
#include <cassert>
#include <memory>

using std::experimental::optional;

struct X
{
};

int main()
{
    static_assert(std::is_assignable<optional<int>, int>::value, "");
    static_assert(std::is_assignable<optional<int>, int&>::value, "");
    static_assert(std::is_assignable<optional<int>&, int>::value, "");
    static_assert(std::is_assignable<optional<int>&, int&>::value, "");
    static_assert(std::is_assignable<optional<int>&, const int&>::value, "");
    static_assert(!std::is_assignable<const optional<int>&, const int&>::value, "");
    static_assert(!std::is_assignable<optional<int>, X>::value, "");
    {
        optional<int> opt;
        opt = 1;
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 1);
    }
    {
        optional<int> opt;
        const int i = 2;
        opt = i;
        assert(static_cast<bool>(opt) == true);
        assert(*opt == i);
    }
    {
        optional<int> opt(3);
        const int i = 2;
        opt = i;
        assert(static_cast<bool>(opt) == true);
        assert(*opt == i);
    }
    {
        optional<std::unique_ptr<int>> opt;
        opt = std::unique_ptr<int>(new int(3));
        assert(static_cast<bool>(opt) == true);
        assert(**opt == 3);
    }
    {
        optional<std::unique_ptr<int>> opt(std::unique_ptr<int>(new int(2)));
        opt = std::unique_ptr<int>(new int(3));
        assert(static_cast<bool>(opt) == true);
        assert(**opt == 3);
    }
}
