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

// template <class T>
//   constexpr
//   optional<typename decay<T>::type>
//   make_optional(T&& v);

#include <experimental/optional>
#include <string>
#include <memory>
#include <cassert>

int main()
{
    using std::experimental::optional;
    using std::experimental::make_optional;

    {
        optional<int> opt = make_optional(2);
        assert(*opt == 2);
    }
    {
        std::string s("123");
        optional<std::string> opt = make_optional(s);
        assert(*opt == s);
    }
    {
        std::string s("123");
        optional<std::string> opt = make_optional(std::move(s));
        assert(*opt == "123");
        assert(s.empty());
    }
    {
        std::unique_ptr<int> s(new int(3));
        optional<std::unique_ptr<int>> opt = make_optional(std::move(s));
        assert(**opt == 3);
        assert(s == nullptr);
    }
}
