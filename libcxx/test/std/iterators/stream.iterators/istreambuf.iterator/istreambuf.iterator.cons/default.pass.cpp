//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// istreambuf_iterator
//
// istreambuf_iterator() throw();
//
// All specializations of istreambuf_iterator shall have a trivial copy constructor,
//    a constexpr default constructor and a trivial destructor.

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    {
        typedef std::istreambuf_iterator<char> T;
        T it;
        assert(it == T());
#if __cplusplus >= 201103L
        constexpr T it2;
#endif
    }
    {
        typedef std::istreambuf_iterator<wchar_t> T;
        T it;
        assert(it == T());
#if __cplusplus >= 201103L
        constexpr T it2;
#endif
    }
}
