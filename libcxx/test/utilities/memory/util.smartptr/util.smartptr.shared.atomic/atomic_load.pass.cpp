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
// shared_ptr<T>
// atomic_load(const shared_ptr<T>* p)

#include <memory>
#include <cassert>

int main()
{
#if __has_feature(cxx_atomic)
    {
        std::shared_ptr<int> p(new int(3));
        std::shared_ptr<int> q = std::atomic_load(&p);
        assert(*q == *p);
    }
#endif
}
