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

// template <class T> struct hash<optional<T>>;

#include <experimental/optional>
#include <string>
#include <memory>
#include <cassert>


int main()
{
    using std::experimental::optional;

    {
        typedef int T;
        optional<T> opt;
        assert(std::hash<optional<T>>{}(opt) == 0);
        opt = 2;
        assert(std::hash<optional<T>>{}(opt) == std::hash<T>{}(*opt));
    }
    {
        typedef std::string T;
        optional<T> opt;
        assert(std::hash<optional<T>>{}(opt) == 0);
        opt = std::string("123");
        assert(std::hash<optional<T>>{}(opt) == std::hash<T>{}(*opt));
    }
    {
        typedef std::unique_ptr<int> T;
        optional<T> opt;
        assert(std::hash<optional<T>>{}(opt) == 0);
        opt = std::unique_ptr<int>(new int(3));
        assert(std::hash<optional<T>>{}(opt) == std::hash<T>{}(*opt));
    }
}
