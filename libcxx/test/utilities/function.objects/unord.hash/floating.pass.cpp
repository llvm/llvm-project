//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <functional>
#include <cassert>
#include <type_traits>
#include <limits>
#include <cmath>

template <class T>
void
test()
{
    static_assert((std::is_base_of<std::unary_function<T, std::size_t>,
                                   std::hash<T> >::value), "");
    std::hash<T> h;
    std::size_t t0 = h(0.);
    std::size_t tn0 = h(-0.);
    std::size_t tp1 = h(0.1);
    std::size_t t1 = h(1);
    std::size_t tn1 = h(-1);
    std::size_t pinf = h(INFINITY);
    std::size_t ninf = h(-INFINITY);
    assert(t0 == tn0);
    assert(t0 != tp1);
    assert(t0 != t1);
    assert(t0 != tn1);
    assert(t0 != pinf);
    assert(t0 != ninf);

    assert(tp1 != t1);
    assert(tp1 != tn1);
    assert(tp1 != pinf);
    assert(tp1 != ninf);

    assert(t1 != tn1);
    assert(t1 != pinf);
    assert(t1 != ninf);

    assert(tn1 != pinf);
    assert(tn1 != ninf);

    assert(pinf != ninf);
}

int main()
{
    test<float>();
    test<double>();
    test<long double>();
}
