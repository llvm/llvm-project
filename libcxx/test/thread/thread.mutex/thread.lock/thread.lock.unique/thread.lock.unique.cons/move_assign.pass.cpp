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

// unique_lock& operator=(unique_lock&& u);

#include <mutex>
#include <cassert>

std::mutex m0;
std::mutex m1;

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::unique_lock<std::mutex> lk0(m0);
    std::unique_lock<std::mutex> lk1(m1);
    lk1 = std::move(lk0);
    assert(lk1.mutex() == &m0);
    assert(lk1.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
