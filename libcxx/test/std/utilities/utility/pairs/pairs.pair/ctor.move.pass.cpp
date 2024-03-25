//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// template <class T1, class T2> struct pair

// pair(pair&&) = default;

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct Dummy {
  Dummy(Dummy const&) = delete;
  Dummy(Dummy &&) = default;
};

struct NotCopyOrMoveConstructible {
  NotCopyOrMoveConstructible() = default;
  NotCopyOrMoveConstructible(NotCopyOrMoveConstructible const&) = delete;
  NotCopyOrMoveConstructible(NotCopyOrMoveConstructible&&) = delete;
};

int main(int, char**)
{
    {
        typedef std::pair<int, short> P1;
        static_assert(std::is_move_constructible<P1>::value, "");
        P1 p1(3, static_cast<short>(4));
        P1 p2 = std::move(p1);
        assert(p2.first == 3);
        assert(p2.second == 4);
    }
    {
        using P = std::pair<Dummy, int>;
        static_assert(!std::is_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
    }
    {
        // When constructing a pair containing a reference, we only bind the
        // reference, so it doesn't matter whether the type is or isn't
        // copy/move constructible.
        {
            using P = std::pair<NotCopyOrMoveConstructible&, int>;
            static_assert(std::is_move_constructible<P>::value, "");

            NotCopyOrMoveConstructible obj;
            P p2{obj, 3};
            P p1(std::move(p2));
            assert(&p1.first == &obj);
            assert(&p2.first == &obj);
        }
        {
            using P = std::pair<NotCopyOrMoveConstructible&&, int>;
            static_assert(std::is_move_constructible<P>::value, "");

            NotCopyOrMoveConstructible obj;
            P p2{std::move(obj), 3};
            P p1(std::move(p2));
            assert(&p1.first == &obj);
            assert(&p2.first == &obj);
        }
    }

    return 0;
}
