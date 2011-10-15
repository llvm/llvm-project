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
//   tuple(allocator_arg_t, const Alloc& a, const tuple<UTypes...>&);

#include <tuple>
#include <cassert>

#include "../allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

int main()
{
    {
        typedef std::tuple<double> T0;
        typedef std::tuple<int> T1;
        T0 t0(2.5);
        T1 t1(std::allocator_arg, A1<int>(), t0);
        assert(std::get<0>(t1) == 2);
    }
    {
        typedef std::tuple<int> T0;
        typedef std::tuple<alloc_first> T1;
        T0 t0(2);
        alloc_first::allocator_constructed = false;
        T1 t1(std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed);
        assert(std::get<0>(t1) == 2);
    }
    {
        typedef std::tuple<int, int> T0;
        typedef std::tuple<alloc_first, alloc_last> T1;
        T0 t0(2, 3);
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        T1 t1(std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed);
        assert(alloc_last::allocator_constructed);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == 3);
    }
    {
        typedef std::tuple<double, int, int> T0;
        typedef std::tuple<int, alloc_first, alloc_last> T1;
        T0 t0(1.5, 2, 3);
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        T1 t1(std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed);
        assert(alloc_last::allocator_constructed);
        assert(std::get<0>(t1) == 1);
        assert(std::get<1>(t1) == 2);
        assert(std::get<2>(t1) == 3);
    }
}
