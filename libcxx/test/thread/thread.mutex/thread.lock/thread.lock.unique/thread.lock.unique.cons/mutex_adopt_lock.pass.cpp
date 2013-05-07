//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock(mutex_type& m, adopt_lock_t);

#include <mutex>
#include <cassert>

int main()
{
    std::mutex m;
    m.lock();
    std::unique_lock<std::mutex> lk(m, std::adopt_lock);
    assert(lk.mutex() == &m);
    assert(lk.owns_lock() == true);
}
