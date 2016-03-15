//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock(unique_lock&& u);

#include <mutex>
#include <cassert>
#include "nasty_containers.hpp"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    typedef std::mutex M;
    M m;
    std::unique_lock<M> lk0(m);
    std::unique_lock<M> lk = std::move(lk0);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }
    {
    typedef nasty_mutex M;
    M m;
    std::unique_lock<M> lk0(m);
    std::unique_lock<M> lk = std::move(lk0);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
