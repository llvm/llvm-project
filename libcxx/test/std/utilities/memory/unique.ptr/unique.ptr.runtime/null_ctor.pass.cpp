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

// The deleter is not called if get() == 0

#include <memory>
#include <cassert>

#include "test_macros.h"

class Deleter
{
    int state_;

    Deleter(Deleter&);
    Deleter& operator=(Deleter&);

public:
    Deleter() : state_(0) {}

    int state() const {return state_;}

    void operator()(void*) {++state_;}
};

int main()
{
    Deleter d;
    assert(d.state() == 0);
    {
    std::unique_ptr<int[], Deleter&> p(nullptr, d);
    assert(p.get() == 0);
    assert(&p.get_deleter() == &d);
    }
#if defined(_LIBCPP_VERSION)
    {
    // The standard only requires the constructor accept nullptr, but libc++
    // also supports the literal 0.
    std::unique_ptr<int[], Deleter&> p(0, d);
    assert(p.get() == 0);
    assert(&p.get_deleter() == &d);
    }
#endif
    assert(d.state() == 0);
}
