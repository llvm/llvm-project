//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t k>
// class shuffle_order_engine
// {
// public:
//     // types
//     typedef typename Engine::result_type result_type;
//
//     // engine characteristics
//     static constexpr size_t table_size = k;
//     static constexpr result_type min() { return Engine::min; }
//     static constexpr result_type max() { return Engine::max; }

#include <random>
#include <type_traits>
#include <cassert>

template <class _Tp>
void where(const _Tp &) {}

void
test1()
{
    typedef std::knuth_b E;
    static_assert(E::table_size == 256, "");
    /*static_*/assert((E::min() == 1)/*, ""*/);
    /*static_*/assert((E::max() == 2147483646)/*, ""*/);
    where(E::table_size);
}

int main()
{
    test1();
}
