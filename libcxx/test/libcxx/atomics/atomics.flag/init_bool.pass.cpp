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

// <atomic>

// struct atomic_flag

// TESTING EXTENSION atomic_flag(bool)

#include <atomic>
#include <cassert>

 // Ensure that static initialization happens; this is PR#37226
extern std::atomic_flag global;
struct X { X() { global.test_and_set(); }};
X x;
std::atomic_flag global = ATOMIC_FLAG_INIT;

int main()
{
    assert(global.test_and_set() == 1);
    {
        std::atomic_flag f(false);
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f(true);
        assert(f.test_and_set() == 1);
    }
}
