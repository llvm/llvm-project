//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr default ctor

// default unique_ptr ctor should only require default Deleter ctor

#include <memory>
#include <cassert>

class Deleter
{
    int state_;

    Deleter(Deleter&);
    Deleter& operator=(Deleter&);

public:
    Deleter() : state_(5) {}

    int state() const {return state_;}

    void operator()(void*) {}
};

int main()
{
    {
    std::unique_ptr<int[]> p;
    assert(p.get() == 0);
    }
    {
    std::unique_ptr<int[], Deleter> p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 5);
    }
}
