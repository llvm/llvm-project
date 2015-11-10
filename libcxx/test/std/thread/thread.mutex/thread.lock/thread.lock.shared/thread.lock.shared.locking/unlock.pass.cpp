//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// XFAIL: libcpp-no-exceptions
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03, c++11

// <shared_mutex>

// template <class Mutex> class shared_lock;

// void unlock();

#include <shared_mutex>
#include <cassert>

bool unlock_called = false;

struct mutex
{
    void lock_shared() {}
    void unlock_shared() {unlock_called = true;}
};

mutex m;

int main()
{
    std::shared_lock<mutex> lk(m);
    lk.unlock();
    assert(unlock_called == true);
    assert(lk.owns_lock() == false);
    try
    {
        lk.unlock();
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
    lk.release();
    try
    {
        lk.unlock();
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
}
