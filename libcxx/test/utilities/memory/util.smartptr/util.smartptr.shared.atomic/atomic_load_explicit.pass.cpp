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
// atomic_load_explicit(const shared_ptr<T>* p, memory_order mo)

#include <memory>
#include <cassert>

int main()
{
#if __has_feature(cxx_atomic)
    {
        const std::shared_ptr<int> p(new int(3));
        std::shared_ptr<int> q = std::atomic_load_explicit(&p, std::memory_order_relaxed);
        assert(*q == *p);
    }
#endif
}
