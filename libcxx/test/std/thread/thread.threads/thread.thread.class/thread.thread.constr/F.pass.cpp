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

// <thread>

// class thread

// template <class F, class ...Args> thread(F&& f, Args&&... args);

// UNSUPPORTED: sanitizer-new-delete

#include <thread>
#include <new>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

unsigned throw_one = 0xFFFF;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    if (throw_one == 0)
        throw std::bad_alloc();
    --throw_one;
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    std::free(p);
}

bool f_run = false;

void f()
{
    f_run = true;
}

class G
{
    int alive_;
public:
    static int n_alive;
    static bool op_run;

    G() : alive_(1) {++n_alive;}
    G(const G& g) : alive_(g.alive_) {++n_alive;}
    ~G() {alive_ = 0; --n_alive;}

    void operator()()
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        op_run = true;
    }

    void operator()(int i, double j)
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        assert(i == 5);
        assert(j == 5.5);
        op_run = true;
    }
};

int G::n_alive = 0;
bool G::op_run = false;

#if TEST_STD_VER >= 11

class MoveOnly
{
    MoveOnly(const MoveOnly&);
public:
    MoveOnly() {}
    MoveOnly(MoveOnly&&) {}

    void operator()(MoveOnly&&)
    {
    }
};

#endif

int main()
{
    {
        std::thread t(f);
        t.join();
        assert(f_run == true);
    }
    f_run = false;
    {
        try
        {
            throw_one = 0;
            std::thread t(f);
            assert(false);
        }
        catch (...)
        {
            throw_one = 0xFFFF;
            assert(!f_run);
        }
    }
    {
        assert(G::n_alive == 0);
        assert(!G::op_run);
        std::thread t((G()));
        t.join();
        assert(G::n_alive == 0);
        assert(G::op_run);
    }
    G::op_run = false;
    {
        try
        {
            throw_one = 0;
            assert(G::n_alive == 0);
            assert(!G::op_run);
            std::thread t((G()));
            assert(false);
        }
        catch (...)
        {
            throw_one = 0xFFFF;
            assert(G::n_alive == 0);
            assert(!G::op_run);
        }
    }
#if TEST_STD_VER >= 11
    {
        assert(G::n_alive == 0);
        assert(!G::op_run);
        std::thread t(G(), 5, 5.5);
        t.join();
        assert(G::n_alive == 0);
        assert(G::op_run);
    }
    {
        std::thread t = std::thread(MoveOnly(), MoveOnly());
        t.join();
    }
#endif
}
