//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(pair<U, V>&& p);

#include <utility>
#include <memory>
#include <cassert>

struct Base
{
    virtual ~Base() {}
};

struct Derived
    : public Base
{
};

int main()
{
    {
        typedef std::pair<std::unique_ptr<Derived>, short> P1;
        typedef std::pair<std::unique_ptr<Base>, long> P2;
        P1 p1(std::unique_ptr<Derived>(), static_cast<short>(4));
        P2 p2;
        p2 = std::move(p1);
        assert(p2.first == nullptr);
        assert(p2.second == 4);
    }
}
