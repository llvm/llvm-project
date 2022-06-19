//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T, class A, class... Args>
// shared_ptr<T> allocate_shared(const A& a, Args&&... args); // T is not an array

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

#include "min_allocator.h"
#include "operator_hijacker.h"
#include "test_allocator.h"
#include "test_macros.h"

int new_count = 0;

struct A
{
    static int count;

    A(int i, char c) : int_(i), char_(c) {++count;}
    A(const A& a)
        : int_(a.int_), char_(a.char_)
        {++count;}
    ~A() {--count;}

    int get_int() const {return int_;}
    char get_char() const {return char_;}

    A* operator& () = delete;
private:
    int int_;
    char char_;
};

int A::count = 0;

struct Zero
{
    static int count;
    Zero() {++count;}
    Zero(Zero const &) {++count;}
    ~Zero() {--count;}
};

int Zero::count = 0;

struct One
{
    static int count;
    int value;
    explicit One(int v) : value(v) {++count;}
    One(One const & o) : value(o.value) {++count;}
    ~One() {--count;}
};

int One::count = 0;


struct Two
{
    static int count;
    int value;
    Two(int v, int) : value(v) {++count;}
    Two(Two const & o) : value(o.value) {++count;}
    ~Two() {--count;}
};

int Two::count = 0;

struct Three
{
    static int count;
    int value;
    Three(int v, int, int) : value(v) {++count;}
    Three(Three const & o) : value(o.value) {++count;}
    ~Three() {--count;}
};

int Three::count = 0;

template<class T>
struct AllocNoConstruct : std::allocator<T>
{
    AllocNoConstruct() = default;

    template <class T1>
    AllocNoConstruct(AllocNoConstruct<T1>) {}

    template <class T1>
    struct rebind {
        typedef AllocNoConstruct<T1> other;
    };

    void construct(void*) { assert(false); }
};

template <class Alloc>
void test()
{
    int const bad = -1;
    {
    std::shared_ptr<Zero> p = std::allocate_shared<Zero>(Alloc());
    assert(Zero::count == 1);
    }
    assert(Zero::count == 0);
    {
    int const i = 42;
    std::shared_ptr<One> p = std::allocate_shared<One>(Alloc(), i);
    assert(One::count == 1);
    assert(p->value == i);
    }
    assert(One::count == 0);
    {
    int const i = 42;
    std::shared_ptr<Two> p = std::allocate_shared<Two>(Alloc(), i, bad);
    assert(Two::count == 1);
    assert(p->value == i);
    }
    assert(Two::count == 0);
    {
    int const i = 42;
    std::shared_ptr<Three> p = std::allocate_shared<Three>(Alloc(), i, bad, bad);
    assert(Three::count == 1);
    assert(p->value == i);
    }
    assert(Three::count == 0);
}

int main(int, char**)
{
    test<bare_allocator<void> >();
    test<test_allocator<void> >();

    test_allocator_statistics alloc_stats;
    {
    int i = 67;
    char c = 'e';
    std::shared_ptr<A> p = std::allocate_shared<A>(test_allocator<A>(54, &alloc_stats), i, c);
    assert(alloc_stats.alloc_count == 1);
    assert(A::count == 1);
    assert(p->get_int() == 67);
    assert(p->get_char() == 'e');
    }
    assert(A::count == 0);
    assert(alloc_stats.alloc_count == 0);
    {
    int i = 67;
    char c = 'e';
    std::shared_ptr<A> p = std::allocate_shared<A>(min_allocator<void>(), i, c);
    assert(A::count == 1);
    assert(p->get_int() == 67);
    assert(p->get_char() == 'e');
    }
    assert(A::count == 0);
    {
    int i = 68;
    char c = 'f';
    std::shared_ptr<A> p = std::allocate_shared<A>(bare_allocator<void>(), i, c);
    assert(A::count == 1);
    assert(p->get_int() == 68);
    assert(p->get_char() == 'f');
    }
    assert(A::count == 0);

    // Make sure std::allocate_shared handles badly-behaved types properly
    {
        std::shared_ptr<operator_hijacker> p1 = std::allocate_shared<operator_hijacker>(min_allocator<operator_hijacker>());
        std::shared_ptr<operator_hijacker> p2 = std::allocate_shared<operator_hijacker>(min_allocator<operator_hijacker>(), operator_hijacker());
        assert(p1 != nullptr);
        assert(p2 != nullptr);
    }

    // Test that we don't call construct before C++20.
#if TEST_STD_VER < 20
    {
    (void)std::allocate_shared<int>(AllocNoConstruct<int>());
    }
#endif // TEST_STD_VER < 20

  return 0;
}
