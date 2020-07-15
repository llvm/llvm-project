//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::function support for the "blocks" extension

// UNSUPPORTED: c++03

// This test requires the Blocks runtime, which is (only?) available
// on Darwin out-of-the-box.
// REQUIRES: has-fblocks && darwin

// RUN: %{build} -fblocks
// RUN: %{run}

#include <functional>
#include <cstdlib>
#include <cassert>

#include <Block.h>

#include "test_macros.h"
#include "count_new.h"


struct A {
  static int count;
  int id_;
  explicit A(int id) { ++count; id_ = id; }
  A(const A &a) { id_ = a.id_; ++count; }
  ~A() { id_ = -1; --count; }
  int operator()() const { return -1; }
  int operator()(int i) const { return i; }
  int operator()(int, int) const { return -2; }
  int operator()(int, int, int) const { return -3; }
  int id() const { return id_; }
};

int A::count = 0;

int g(int) { return 0; }

int main(int, char**)
{
    // swap
    {
        std::function<int(int)> f1 = g;
        std::function<int(int)> f2 = ^(int x) { return x + 1; };
        assert(globalMemCounter.checkOutstandingNewEq(0));
        RTTI_ASSERT(*f1.target<int(*)(int)>() == g);
        RTTI_ASSERT(*f2.target<int(^)(int)>() != 0);
        swap(f1, f2);
        assert(globalMemCounter.checkOutstandingNewEq(0));
        RTTI_ASSERT(*f1.target<int(^)(int)>() != 0);
        RTTI_ASSERT(*f2.target<int(*)(int)>() == g);
    }

    // operator bool
    {
        std::function<int(int)> f;
        assert(!f);
        f = ^(int x) { return x+1; };
        assert(f);
    }

    // operator()
    {
        std::function<int ()> r1(^{ return 4; });
        assert(r1() == 4);
    }
    {
        __block bool called = false;
        std::function<void ()> r1(^{ called = true; });
        r1();
        assert(called);
    }
    {
        __block int param = 0;
        std::function<void (int)> r1(^(int x){ param = x; });
        r1(4);
        assert(param == 4);
    }
    {
        std::function<int (int)> r1(^(int x){ return x + 4; });
        assert(r1(3) == 7);
    }
    {
        __block int param1 = 0;
        __block int param2 = 0;
        std::function<void (int, int)> r1(^(int x, int y){ param1 = x; param2 = y; });
        r1(3, 4);
        assert(param1 == 3);
        assert(param2 == 4);
    }
    {
        std::function<int (int, int)> r1(^(int x, int y){ return x + y; });
        assert(r1(3, 4) == 7);
    }

    // swap
    {
        std::function<int(int)> f1 = A(999);
        std::function<int(int)> f2 = ^(int x) { return x + 1; };
        assert(A::count == 1);
        assert(globalMemCounter.checkOutstandingNewEq(1));
        RTTI_ASSERT(f1.target<A>()->id() == 999);
        RTTI_ASSERT((*f2.target<int(^)(int)>())(13) == 14);
        f1.swap(f2);
        assert(A::count == 1);
        assert(globalMemCounter.checkOutstandingNewEq(1));
        RTTI_ASSERT((*f1.target<int(^)(int)>())(13) == 14);
        RTTI_ASSERT(f2.target<A>()->id() == 999);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(A::count == 0);

    // operator== and operator!=
    {
        std::function<int(int)> f;
        assert(f == nullptr);
        assert(nullptr == f);
        f = ^(int x) { return x + 1; };
        assert(f != nullptr);
        assert(nullptr != f);
    }

    // target
    {
        int (^block)(int) = Block_copy(^(int x) { return x + 1; });
        std::function<int(int)> f = block;
        RTTI_ASSERT(*f.target<int(^)(int)>() == block);
        RTTI_ASSERT(f.target<int(*)(int)>() == 0);
        Block_release(block);
    }

    // target_type
    {
        std::function<int(int)> f = ^(int x) { return x + 1; };
        RTTI_ASSERT(f.target_type() == typeid(int(^)(int)));
    }

    return 0;
}
