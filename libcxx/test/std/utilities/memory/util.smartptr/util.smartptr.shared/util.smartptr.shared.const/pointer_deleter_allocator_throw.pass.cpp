//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// <memory>

// template<class Y, class D, class A> shared_ptr(Y* p, D d, A a);

#include <memory>
#include <cassert>
#include "test_macros.h"
#include "deleter_types.h"
#include "test_allocator.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    test_allocator_statistics alloc_stats;
    A* ptr = new A;
    try
    {
        alloc_stats.throw_after = 0;
        std::shared_ptr<A> p(ptr, test_deleter<A>(3), test_allocator<A>(5, &alloc_stats));
        assert(false);
    }
    catch (std::bad_alloc&)
    {
        assert(alloc_stats.count == 0);
        assert(alloc_stats.count == 0);
        assert(test_deleter<A>::dealloc_count == 1);
        assert(alloc_stats.count == 0);
        assert(alloc_stats.alloc_count == 0);
    }

  return 0;
}
