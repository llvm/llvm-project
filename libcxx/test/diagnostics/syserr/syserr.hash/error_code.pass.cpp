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

#include <system_error>
#include <cassert>
#include <type_traits>

void
test(int i)
{
    typedef std::error_code T;
    typedef std::hash<T> H;
    static_assert((std::is_base_of<std::unary_function<T, std::size_t>,
                                   H>::value), "");
    H h;
    T ec(i, std::system_category());
    assert(h(ec) == i);
}

int main()
{
    test(0);
    test(2);
    test(10);
}
