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

// template <class Alloc, class... UTypes>
//   tuple(allocator_arg_t, const Alloc& a, UTypes&&...);

#include <tuple>
#include <cassert>

#include "../MoveOnly.h"
#include "../allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

int main()
{
    {
        std::tuple<MoveOnly> t(std::allocator_arg, A1<int>(), MoveOnly(0));
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                         MoveOnly(0), MoveOnly(1));
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
    }
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                                   MoveOnly(0),
                                                   1, 2);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == 2);
    }
    {
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        std::tuple<int, alloc_first, alloc_last> t(std::allocator_arg,
                                                   A1<int>(5), 1, 2, 3);
        assert(std::get<0>(t) == 1);
        assert(alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first(2));
        assert(alloc_last::allocator_constructed);
        assert(std::get<2>(t) == alloc_last(3));
    }
    // extensions
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                                   0, 1);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == MoveOnly());
    }
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                                   0);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == MoveOnly());
        assert(std::get<2>(t) == MoveOnly());
    }
}
