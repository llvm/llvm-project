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

template <class T>
void
test()
{
    static_assert((std::is_base_of<std::unary_function<T, std::size_t>,
                                   std::hash<T> >::value), "");
    std::hash<T> h;
    typedef typename std::remove_pointer<T>::type type;
    type i;
    type j;
    assert(h(&i) != h(&j));
}

int main()
{
    test<int*>();
}
