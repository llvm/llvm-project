//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++experimental
// UNSUPPORTED: c++98, c++03

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// EXTENSION
// std::size_t polymorphic_allocator<T>::max_size() const noexcept

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_memory_resource.hpp"

namespace ex = std::experimental::pmr;

template <std::size_t S>
std::size_t getMaxSize() {
    using T = typename std::aligned_storage<S>::type;
    static_assert(sizeof(T) == S, "Required for test");
    return ex::polymorphic_allocator<T>{}.max_size();
}

template <std::size_t S, std::size_t A>
std::size_t getMaxSize() {
    using T = typename std::aligned_storage<S, A>::type;
    static_assert(sizeof(T) == S, "Required for test");
    return ex::polymorphic_allocator<T>{}.max_size();
}

int main()
{
    {
        using Alloc = ex::polymorphic_allocator<int>;
        using Traits = std::allocator_traits<Alloc>;
        const Alloc a;
        static_assert(std::is_same<decltype(a.max_size()), Traits::size_type>::value, "");
        static_assert(noexcept(a.max_size()), "");
    }
    {
        constexpr std::size_t Max = std::numeric_limits<std::size_t>::max();
        assert(getMaxSize<1>()    == Max);
        assert(getMaxSize<2>()    == Max / 2);
        assert(getMaxSize<4>()    == Max / 4);
        assert(getMaxSize<8>()    == Max / 8);
        assert(getMaxSize<16>()   == Max / 16);
        assert(getMaxSize<32>()   == Max / 32);
        assert(getMaxSize<64>()   == Max / 64);
        assert(getMaxSize<1024>() == Max / 1024);

        assert((getMaxSize<6,  2>() == Max / 6));
        assert((getMaxSize<12, 4>() == Max / 12));
    }
}
