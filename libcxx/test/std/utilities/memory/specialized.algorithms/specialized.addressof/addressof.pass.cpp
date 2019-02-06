//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <ObjectType T> T* addressof(T& r);

#include <memory>
#include <cassert>

struct A
{
    void operator&() const {}
};

struct nothing {
    operator char&()
    {
        static char c;
        return c;
    }
};

int main()
{
    {
    int i;
    double d;
    assert(std::addressof(i) == &i);
    assert(std::addressof(d) == &d);
    A* tp = new A;
    const A* ctp = tp;
    assert(std::addressof(*tp) == tp);
    assert(std::addressof(*ctp) == tp);
    delete tp;
    }
    {
    union
    {
        nothing n;
        int i;
    };
    assert(std::addressof(n) == (void*)std::addressof(i));
    }
}
