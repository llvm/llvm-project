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
// bool
// atomic_compare_exchange_weak(shared_ptr<T>* p, shared_ptr<T>* v,
//                              shared_ptr<T> w);

#include <memory>
#include <cassert>

int main()
{
#if __has_feature(cxx_atomic)
    {
        std::shared_ptr<int> p(new int(4));
        std::shared_ptr<int> v(new int(3));
        std::shared_ptr<int> w(new int(2));
        bool b = std::atomic_compare_exchange_weak(&p, &v, w);
        assert(b == false);
        assert(*p == 4);
        assert(*v == 4);
        assert(*w == 2);
    }
    {
        std::shared_ptr<int> p(new int(4));
        std::shared_ptr<int> v = p;
        std::shared_ptr<int> w(new int(2));
        bool b = std::atomic_compare_exchange_weak(&p, &v, w);
        assert(b == true);
        assert(*p == 2);
        assert(*v == 4);
        assert(*w == 2);
    }
#endif
}
