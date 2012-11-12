//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template <class T>
// shared_ptr<T>
// atomic_exchange(shared_ptr<T>* p, shared_ptr<T> r)

#include <memory>
#include <cassert>

int main()
{
#if __has_feature(cxx_atomic)
    {
        std::shared_ptr<int> p(new int(4));
        std::shared_ptr<int> r(new int(3));
        r = std::atomic_exchange(&p, r);
        assert(*p == 3);
        assert(*r == 4);
    }
#endif
}
