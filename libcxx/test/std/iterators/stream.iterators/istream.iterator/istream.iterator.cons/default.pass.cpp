//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// constexpr istream_iterator();

#include <iterator>
#include <cassert>

int main()
{
    {
    typedef std::istream_iterator<int> T;
    T it;
    assert(it == T());
#if __cplusplus >= 201103L
    constexpr T it2;
#endif
    }

}
