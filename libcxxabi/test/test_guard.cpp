//===----------------------------- test_guard.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "cxxabi.h"

#include <cassert>
#include <thread>

// Ensure that we initialize each variable once and only once.
namespace test1 {
    static int run_count = 0;
    int increment() {
        ++run_count;
        return 0;
    }
    void helper() {
        static int a = increment();
    }
    void test() {
        static int a = increment();
        assert(run_count == 1);
        static int b = increment();
        assert(run_count == 2);
        helper();
        assert(run_count == 3);
        helper();
        assert(run_count == 3);
    }
}

// When initialization fails, ensure that we try to initialize it again next
// time.
namespace test2 {
    static int run_count = 0;
    int increment() {
        ++run_count;
        throw 0;
    }
    void helper() {
        try {
            static int a = increment();
            assert(0);
        } catch (...) {}
    }
    void test() {
        helper();
        assert(run_count == 1);
        helper();
        assert(run_count == 2);
    }
}

// Check that we can initialize a second value while initializing a first.
namespace test3 {
    int zero() {
        return 0;
    }

    int one() {
        static int b = zero();
        return 0;
    }

    void test() {
        static int a = one();
    }
}

// A simple thread test of two threads racing to initialize a variable. This
// isn't guaranteed to catch any particular threading problems.
namespace test4 {
    static int run_count = 0;
    int increment() {
        ++run_count;
        return 0;
    }

    void helper() {
        static int a = increment();
    }

    void test() {
        std::thread t1(helper), t2(helper);
        t1.join();
        t2.join();
        assert(run_count == 1);
    }
}

// Check that we don't re-initialize a static variable even when it's
// encountered from two different threads.
namespace test5 {
    static int run_count = 0;
    int zero() {
        ++run_count;
        return 0;
    }

    int one() {
        static int b = zero();
        return 0;
    }

    void another_helper() {
        static int a = one();
    }

    void helper() {
        static int a = one();
        std::thread t(another_helper);
        t.join();
    }

    void test() {
        std::thread t(helper);
        t.join();
        assert(run_count == 1);
    }
}

int main()
{
    test1::test();
    test2::test();
    test3::test();
    test4::test();
    test5::test();
}
