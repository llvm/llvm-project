//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This test uses new symbols that were not defined in the libc++ shipped on
// darwin11 and darwin12:
// XFAIL: with_system_lib=x86_64-apple-darwin11
// XFAIL: with_system_lib=x86_64-apple-darwin12

// <memory>

// shared_ptr

// template <class T>
// void
// atomic_store_explicit(shared_ptr<T>* p, shared_ptr<T> r, memory_order mo)

#include <memory>
#include <cassert>

int main()
{
#if __has_feature(cxx_atomic)
    {
        std::shared_ptr<int> p;
        std::shared_ptr<int> r(new int(3));
        std::atomic_store_explicit(&p, r, std::memory_order_seq_cst);
        assert(*p == *r);
    }
#endif
}
