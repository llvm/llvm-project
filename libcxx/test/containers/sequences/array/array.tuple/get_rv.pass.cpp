//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// template <size_t I, class T, size_t N> T&& get(array<T, N>&& a);

#include <array>
#include <memory>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef std::unique_ptr<double> T;
        typedef std::array<T, 1> C;
        C c = {std::unique_ptr<double>(new double(3.5))};
        T t = std::get<0>(std::move(c));
        assert(*t == 3.5);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
