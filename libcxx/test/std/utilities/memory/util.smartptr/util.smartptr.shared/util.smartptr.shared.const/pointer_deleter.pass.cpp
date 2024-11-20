//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y, class D> shared_ptr(Y* p, D d);

#include <memory>
#include <cassert>
#include "test_macros.h"
#include "deleter_types.h"

#include "types.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

// LWG 3233. Broken requirements for shared_ptr converting constructors
// https://cplusplus.github.io/LWG/issue3233
// https://llvm.org/PR60258
// Invalid constructor SFINAE for std::shared_ptr's array ctors
static_assert( std::is_constructible<std::shared_ptr<int>,  int*, test_deleter<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int>,  int*, bad_deleter>::value, "");
static_assert( std::is_constructible<std::shared_ptr<base>,  derived*, test_deleter<base> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<A>,  int*, test_deleter<A> >::value, "");

#if TEST_STD_VER >= 17
static_assert( std::is_constructible<std::shared_ptr<int[]>,  int*, test_deleter<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[]>,  int*, bad_deleter>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[]>,  int(*)[], test_deleter<int> >::value, "");
static_assert( std::is_constructible<std::shared_ptr<int[5]>, int*, test_deleter<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[5]>, int*, bad_deleter>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[5]>, int(*)[5], test_deleter<int> >::value, "");
#endif

int f() { return 5; }

// https://cplusplus.github.io/LWG/issue3018
// LWG 3018. shared_ptr of function type
struct function_pointer_deleter {
  function_pointer_deleter(bool& deleter_called) : deleter_called_(deleter_called) {}

  void operator()(int (*)()) const { deleter_called_ = true; }

  bool& deleter_called_;
};

void test_function_type() {
  bool deleter_called = false;
  {
    std::shared_ptr<int()> p(&f, function_pointer_deleter(deleter_called));
    assert((*p)() == 5);
  }
  assert(deleter_called);
}

int main(int, char**)
{
    {
        A* ptr = new A;
        std::shared_ptr<A> p(ptr, test_deleter<A>(3));
        assert(A::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
        assert(test_deleter<A>::count == 1);
        assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
        test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
        assert(d);
        assert(d->state() == 3);
#endif
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);

    {
        A const* ptr = new A;
        std::shared_ptr<A const> p(ptr, test_deleter<A const>(3));
        assert(p.get() == ptr);
    }

    {
        // Make sure we can't construct with:
        //    a) a deleter that doesn't have an operator ()(int*)
        //    b) a deleter that doesn't have a move constructor.
        static_assert(!std::is_constructible<std::shared_ptr<int>, int*, bad_deleter>::value, "");
        static_assert(!std::is_constructible<std::shared_ptr<int>, int*, no_move_deleter>::value, "");

        // Make sure that we can construct a shared_ptr where the element type and pointer type
        // aren't "convertible" but are "compatible".
        static_assert(!std::is_constructible<std::shared_ptr<derived[4]>, base[4], test_deleter<derived[4]> >::value, "");
    }

#if TEST_STD_VER >= 11
    {
        move_deleter<int> d(0);
        std::shared_ptr<int> p0(new int, std::move(d));
        std::shared_ptr<int> p1(nullptr, std::move(d));
    }
#endif // TEST_STD_VER >= 11

#if TEST_STD_VER >= 14
    {
      // LWG 4110
      auto deleter = [](auto pointer) { delete pointer; };
      std::shared_ptr<int> p(new int, deleter);
    }

    {
      std::shared_ptr<int> p(NULL, [](auto){});
    }
#endif

#if TEST_STD_VER >= 17
    {
      // See https://github.com/llvm/llvm-project/pull/93071#issuecomment-2166047398
      std::shared_ptr<char[]> a(new char[10], std::default_delete<char[]>());
    }
#endif

  test_function_type();
  return 0;
}
