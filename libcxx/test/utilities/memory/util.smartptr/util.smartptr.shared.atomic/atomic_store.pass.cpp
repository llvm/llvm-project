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
// void
// atomic_store(shared_ptr<T>* p, shared_ptr<T> r)

#include <memory>
#include <cassert>

int main()
{
#if __has_feature(cxx_atomic)
    {
        std::shared_ptr<int> p;
        std::shared_ptr<int> r(new int(3));
        std::atomic_store(&p, r);
        assert(*p == *r);
    }
#endif
}
