//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     typename tuple_element<I, std::pair<T1, T2> >::type&&
//     get(pair<T1, T2>&&);

#include <utility>
#include <memory>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef std::pair<std::unique_ptr<int>, short> P;
        P p(std::unique_ptr<int>(new int(3)), 4);
        std::unique_ptr<int> ptr = std::get<0>(std::move(p));
        assert(*ptr == 3);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
