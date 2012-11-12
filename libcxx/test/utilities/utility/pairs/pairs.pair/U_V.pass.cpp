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

// template<class U, class V> pair(U&& x, V&& y);

#include <utility>
#include <memory>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef std::pair<std::unique_ptr<int>, short*> P;
        P p(std::unique_ptr<int>(new int(3)), nullptr);
        assert(*p.first == 3);
        assert(p.second == nullptr);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
