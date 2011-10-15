//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

#include <tuple>
#include <cassert>

#include "../MoveOnly.h"

int main()
{
    {
        std::tuple<MoveOnly> t = MoveOnly(0);
    }
}
