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

// template<class T>
// bool
// atomic_is_lock_free(const shared_ptr<T>* p);

#include <memory>
#include <cassert>

int main()
{
#if __has_feature(cxx_atomic)
    {
        const std::shared_ptr<int> p(new int(3));
        assert(std::atomic_is_lock_free(&p) == false);
    }
#endif
}
