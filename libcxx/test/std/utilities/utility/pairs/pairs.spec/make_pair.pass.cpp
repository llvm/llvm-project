//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> pair<V1, V2> make_pair(T1&&, T2&&);

#include <utility>
#include <memory>
#include <cassert>

#include "test_macros.h"

int main()
{
    {
        typedef std::pair<int, short> P1;
        P1 p1 = std::make_pair(3, 4);
        assert(p1.first == 3);
        assert(p1.second == 4);
    }

#if TEST_STD_VER >= 11
    {
        typedef std::pair<std::unique_ptr<int>, short> P1;
        P1 p1 = std::make_pair(std::unique_ptr<int>(new int(3)), 4);
        assert(*p1.first == 3);
        assert(p1.second == 4);
    }
    {
        typedef std::pair<std::unique_ptr<int>, short> P1;
        P1 p1 = std::make_pair(nullptr, 4);
        assert(p1.first == nullptr);
        assert(p1.second == 4);
    }
#endif
#if TEST_STD_VER >= 14
    {
        typedef std::pair<int, short> P1;
        constexpr P1 p1 = std::make_pair(3, 4);
        static_assert(p1.first == 3, "");
        static_assert(p1.second == 4, "");
    }
#endif

}
