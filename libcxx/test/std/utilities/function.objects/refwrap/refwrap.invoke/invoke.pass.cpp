//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

//  template <class... ArgTypes>
//    constexpr typename result_of<T&(ArgTypes&&...)>::type               // constexpr since C++20
//        operator() (ArgTypes&&...) const
//            noexcept(is_nothrow_invocable_v<T&, ArgTypes...>);          // noexcept since C++17

#include <functional>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER > 14
#  define INVOKE_NOEXCEPT(expected, ...) static_assert(noexcept(__VA_ARGS__) == expected)
#else
#  define INVOKE_NOEXCEPT(expected, ...)
#endif

int count = 0;

// 1 arg, return void

void f_void_1(int i)
{
    count += i;
}

struct A_void_1
{
    void operator()(int i)
    {
        count += i;
    }

    void mem1() {++count;}
    void mem2() const {++count;}
};

void
test_void_1()
{
    int save_count = count;
    // function
    {
    std::reference_wrapper<void (int)> r1(f_void_1);
    int i = 2;
    r1(i);
    INVOKE_NOEXCEPT(false, r1(i));
    assert(count == save_count+2);
    save_count = count;
    }
    // function pointer
    {
    void (*fp)(int) = f_void_1;
    std::reference_wrapper<void (*)(int)> r1(fp);
    int i = 3;
    r1(i);
    INVOKE_NOEXCEPT(false, r1(i));
    assert(count == save_count+3);
    save_count = count;
    }
    // functor
    {
    A_void_1 a0;
    std::reference_wrapper<A_void_1> r1(a0);
    int i = 4;
    r1(i);
    INVOKE_NOEXCEPT(false, r1(i));
    assert(count == save_count+4);
    save_count = count;
    }
    // member function pointer
    {
    void (A_void_1::*fp)() = &A_void_1::mem1;
    std::reference_wrapper<void (A_void_1::*)()> r1(fp);
    A_void_1 a;
    r1(a);
    INVOKE_NOEXCEPT(false, r1(a));
    assert(count == save_count+1);
    save_count = count;
    A_void_1* ap = &a;
    r1(ap);
    INVOKE_NOEXCEPT(false, r1(ap));
    assert(count == save_count+1);
    save_count = count;
    }
    // const member function pointer
    {
    void (A_void_1::*fp)() const = &A_void_1::mem2;
    std::reference_wrapper<void (A_void_1::*)() const> r1(fp);
    A_void_1 a;
    r1(a);
    INVOKE_NOEXCEPT(false, r1(a));
    assert(count == save_count+1);
    save_count = count;
    A_void_1* ap = &a;
    r1(ap);
    INVOKE_NOEXCEPT(false, r1(ap));
    assert(count == save_count+1);
    save_count = count;
    }
}

// 1 arg, return int

int f_int_1(int i)
{
    return i + 1;
}

struct A_int_1
{
    A_int_1() : data_(5) {}
    int operator()(int i)
    {
        return i - 1;
    }

    int mem1() {return 3;}
    int mem2() const {return 4;}
    int data_;
};

void
test_int_1()
{
    // function
    {
    std::reference_wrapper<int (int)> r1(f_int_1);
    int i = 2;
    assert(r1(i) == 3);
    INVOKE_NOEXCEPT(false, r1(i));
    }
    // function pointer
    {
    int (*fp)(int) = f_int_1;
    std::reference_wrapper<int (*)(int)> r1(fp);
    int i = 3;
    assert(r1(i) == 4);
    INVOKE_NOEXCEPT(false, r1(i));
    }
    // functor
    {
    A_int_1 a0;
    std::reference_wrapper<A_int_1> r1(a0);
    int i = 4;
    assert(r1(i) == 3);
    INVOKE_NOEXCEPT(false, r1(i));
    }
    // member function pointer
    {
    int (A_int_1::*fp)() = &A_int_1::mem1;
    std::reference_wrapper<int (A_int_1::*)()> r1(fp);
    A_int_1 a;
    assert(r1(a) == 3);
    INVOKE_NOEXCEPT(false, r1(a));
    A_int_1* ap = &a;
    assert(r1(ap) == 3);
    INVOKE_NOEXCEPT(false, r1(ap));
    }
    // const member function pointer
    {
    int (A_int_1::*fp)() const = &A_int_1::mem2;
    std::reference_wrapper<int (A_int_1::*)() const> r1(fp);
    A_int_1 a;
    assert(r1(a) == 4);
    INVOKE_NOEXCEPT(false, r1(a));
    A_int_1* ap = &a;
    assert(r1(ap) == 4);
    INVOKE_NOEXCEPT(false, r1(ap));
    }
    // member data pointer
    {
    int A_int_1::*fp = &A_int_1::data_;
    std::reference_wrapper<int A_int_1::*> r1(fp);
    A_int_1 a;
    assert(r1(a) == 5);
    INVOKE_NOEXCEPT(true, r1(a));
    r1(a) = 6;
    assert(r1(a) == 6);
    A_int_1* ap = &a;
    assert(r1(ap) == 6);
    r1(ap) = 7;
    assert(r1(ap) == 7);
    INVOKE_NOEXCEPT(true, r1(ap));
    }
}

// 2 arg, return void

void f_void_2(int i, int j)
{
    count += i+j;
}

struct A_void_2
{
    void operator()(int i, int j)
    {
        count += i+j;
    }

    void mem1(int i) {count += i;}
    void mem2(int i) const {count += i;}
};

void
test_void_2()
{
    int save_count = count;
    // function
    {
    std::reference_wrapper<void (int, int)> r1(f_void_2);
    int i = 2;
    int j = 3;
    r1(i, j);
    INVOKE_NOEXCEPT(false, r1(i, j));
    assert(count == save_count+5);
    save_count = count;
    }
    // function pointer
    {
    void (*fp)(int, int) = f_void_2;
    std::reference_wrapper<void (*)(int, int)> r1(fp);
    int i = 3;
    int j = 4;
    r1(i, j);
    INVOKE_NOEXCEPT(false, r1(i, j));
    assert(count == save_count+7);
    save_count = count;
    }
    // functor
    {
    A_void_2 a0;
    std::reference_wrapper<A_void_2> r1(a0);
    int i = 4;
    int j = 5;
    r1(i, j);
    INVOKE_NOEXCEPT(false, r1(i, j));
    assert(count == save_count+9);
    save_count = count;
    }
    // member function pointer
    {
    void (A_void_2::*fp)(int) = &A_void_2::mem1;
    std::reference_wrapper<void (A_void_2::*)(int)> r1(fp);
    A_void_2 a;
    int i = 3;
    r1(a, i);
    assert(count == save_count+3);
    save_count = count;
    A_void_2* ap = &a;
    r1(ap, i);
    INVOKE_NOEXCEPT(false, r1(ap, i));
    assert(count == save_count+3);
    save_count = count;
    }
    // const member function pointer
    {
    void (A_void_2::*fp)(int) const = &A_void_2::mem2;
    std::reference_wrapper<void (A_void_2::*)(int) const> r1(fp);
    A_void_2 a;
    int i = 4;
    r1(a, i);
    INVOKE_NOEXCEPT(false, r1(a, i));
    assert(count == save_count+4);
    save_count = count;
    A_void_2* ap = &a;
    r1(ap, i);
    INVOKE_NOEXCEPT(false, r1(ap, i));
    assert(count == save_count+4);
    save_count = count;
    }
}

// 2 arg, return int

int f_int_2(int i, int j)
{
    return i+j;
}

struct A_int_2
{
    int operator()(int i, int j)
    {
        return i+j;
    }

    int mem1(int i) {return i+1;}
    int mem2(int i) const {return i+2;}
};

void
test_int_2()
{
    // function
    {
    std::reference_wrapper<int (int, int)> r1(f_int_2);
    int i = 2;
    int j = 3;
    assert(r1(i, j) == i+j);
    INVOKE_NOEXCEPT(false, r1(i, j));
    }
    // function pointer
    {
    int (*fp)(int, int) = f_int_2;
    std::reference_wrapper<int (*)(int, int)> r1(fp);
    int i = 3;
    int j = 4;
    assert(r1(i, j) == i+j);
    INVOKE_NOEXCEPT(false, r1(i, j));
    }
    // functor
    {
    A_int_2 a0;
    std::reference_wrapper<A_int_2> r1(a0);
    int i = 4;
    int j = 5;
    assert(r1(i, j) == i+j);
    INVOKE_NOEXCEPT(false, r1(i, j));
    }
    // member function pointer
    {
    int(A_int_2::*fp)(int) = &A_int_2::mem1;
    std::reference_wrapper<int (A_int_2::*)(int)> r1(fp);
    A_int_2 a;
    int i = 3;
    assert(r1(a, i) == i+1);
    INVOKE_NOEXCEPT(false, r1(a, i));
    A_int_2* ap = &a;
    assert(r1(ap, i) == i+1);
    INVOKE_NOEXCEPT(false, r1(ap, i));
    }
    // const member function pointer
    {
    int (A_int_2::*fp)(int) const = &A_int_2::mem2;
    std::reference_wrapper<int (A_int_2::*)(int) const> r1(fp);
    A_int_2 a;
    int i = 4;
    assert(r1(a, i) == i+2);
    INVOKE_NOEXCEPT(false, r1(a, i));
    A_int_2* ap = &a;
    assert(r1(ap, i) == i+2);
    INVOKE_NOEXCEPT(false, r1(ap, i));
    }
}

#if TEST_STD_VER >= 11

// 1 arg, return void, noexcept

void f_void_1_noexcept(int i) noexcept
{
    count += i;
}

struct A_void_1_noexcept
{
    void operator()(int i) noexcept
    {
        count += i;
    }

    void mem1() noexcept {++count;}
    void mem2() const noexcept {++count;}
};

void
test_void_1_noexcept()
{
    int save_count = count;
    // function
    {
    std::reference_wrapper<void (int) noexcept> r1(f_void_1_noexcept);
    int i = 2;
    r1(i);
    INVOKE_NOEXCEPT(true, r1(i));
    assert(count == save_count+2);
    save_count = count;
    }
    // function pointer
    {
    void (*fp)(int) noexcept = f_void_1_noexcept;
    std::reference_wrapper<void (*)(int) noexcept> r1(fp);
    int i = 3;
    r1(i);
    INVOKE_NOEXCEPT(true, r1(i));
    assert(count == save_count+3);
    save_count = count;
    }
    // functor
    {
    A_void_1_noexcept a0;
    std::reference_wrapper<A_void_1_noexcept> r1(a0);
    int i = 4;
    r1(i);
    INVOKE_NOEXCEPT(true, r1(i));
    assert(count == save_count+4);
    save_count = count;
    }
    // member function pointer
    {
    void (A_void_1_noexcept::*fp)() noexcept = &A_void_1_noexcept::mem1;
    std::reference_wrapper<void (A_void_1_noexcept::*)() noexcept> r1(fp);
    A_void_1_noexcept a;
    r1(a);
    INVOKE_NOEXCEPT(true, r1(a));
    assert(count == save_count+1);
    save_count = count;
    A_void_1_noexcept* ap = &a;
    r1(ap);
    INVOKE_NOEXCEPT(true, r1(ap));
    assert(count == save_count+1);
    save_count = count;
    }
    // const member function pointer
    {
    void (A_void_1_noexcept::*fp)() const noexcept = &A_void_1_noexcept::mem2;
    std::reference_wrapper<void (A_void_1_noexcept::*)() const noexcept> r1(fp);
    A_void_1_noexcept a;
    r1(a);
    INVOKE_NOEXCEPT(true, r1(a));
    assert(count == save_count+1);
    save_count = count;
    A_void_1_noexcept* ap = &a;
    r1(ap);
    INVOKE_NOEXCEPT(true, r1(ap));
    assert(count == save_count+1);
    save_count = count;
    }
}

// 1 arg, return int, noexcept

int f_int_1_noexcept(int i) noexcept
{
    return i + 1;
}

struct A_int_1_noexcept
{
    A_int_1_noexcept() : data_(5) {}
    int operator()(int i) noexcept
    {
        return i - 1;
    }

    int mem1() noexcept {return 3;}
    int mem2() const noexcept {return 4;}
    int data_;
};

void
test_int_1_noexcept()
{
    // function
    {
    std::reference_wrapper<int (int) noexcept> r1(f_int_1_noexcept);
    int i = 2;
    assert(r1(i) == 3);
    INVOKE_NOEXCEPT(true, r1(i));
    }
    // function pointer
    {
    int (*fp)(int) noexcept = f_int_1_noexcept;
    std::reference_wrapper<int (*)(int) noexcept> r1(fp);
    int i = 3;
    assert(r1(i) == 4);
    INVOKE_NOEXCEPT(true, r1(i));
    }
    // functor
    {
    A_int_1_noexcept a0;
    std::reference_wrapper<A_int_1_noexcept> r1(a0);
    int i = 4;
    assert(r1(i) == 3);
    INVOKE_NOEXCEPT(true, r1(i));
    }
    // member function pointer
    {
    int (A_int_1_noexcept::*fp)() noexcept = &A_int_1_noexcept::mem1;
    std::reference_wrapper<int (A_int_1_noexcept::*)() noexcept> r1(fp);
    A_int_1_noexcept a;
    assert(r1(a) == 3);
    INVOKE_NOEXCEPT(true, r1(a));
    A_int_1_noexcept* ap = &a;
    assert(r1(ap) == 3);
    INVOKE_NOEXCEPT(true, r1(ap));
    }
    // const member function pointer
    {
    int (A_int_1_noexcept::*fp)() const noexcept = &A_int_1_noexcept::mem2;
    std::reference_wrapper<int (A_int_1_noexcept::*)() const noexcept> r1(fp);
    A_int_1_noexcept a;
    assert(r1(a) == 4);
    INVOKE_NOEXCEPT(true, r1(a));
    A_int_1_noexcept* ap = &a;
    assert(r1(ap) == 4);
    INVOKE_NOEXCEPT(true, r1(ap));
    }
    // member data pointer
    {
    int A_int_1_noexcept::*fp = &A_int_1_noexcept::data_;
    std::reference_wrapper<int A_int_1_noexcept::*> r1(fp);
    A_int_1_noexcept a;
    assert(r1(a) == 5);
    INVOKE_NOEXCEPT(true, r1(a));
    r1(a) = 6;
    assert(r1(a) == 6);
    A_int_1_noexcept* ap = &a;
    assert(r1(ap) == 6);
    r1(ap) = 7;
    assert(r1(ap) == 7);
    INVOKE_NOEXCEPT(true, r1(ap));
    }
}

// 2 arg, return void, noexcept

void f_void_2_noexcept(int i, int j) noexcept
{
    count += i+j;
}

struct A_void_2_noexcept
{
    void operator()(int i, int j) noexcept
    {
        count += i+j;
    }

    void mem1(int i) noexcept {count += i;}
    void mem2(int i) const noexcept {count += i;}
};

void
test_void_2_noexcept()
{
    int save_count = count;
    // function
    {
    std::reference_wrapper<void (int, int) noexcept> r1(f_void_2_noexcept);
    int i = 2;
    int j = 3;
    r1(i, j);
    INVOKE_NOEXCEPT(true, r1(i, j));
    assert(count == save_count+5);
    save_count = count;
    }
    // function pointer
    {
    void (*fp)(int, int) noexcept = f_void_2_noexcept;
    std::reference_wrapper<void (*)(int, int) noexcept> r1(fp);
    int i = 3;
    int j = 4;
    r1(i, j);
    INVOKE_NOEXCEPT(true, r1(i, j));
    assert(count == save_count+7);
    save_count = count;
    }
    // functor
    {
    A_void_2_noexcept a0;
    std::reference_wrapper<A_void_2_noexcept> r1(a0);
    int i = 4;
    int j = 5;
    r1(i, j);
    INVOKE_NOEXCEPT(true, r1(i, j));
    assert(count == save_count+9);
    save_count = count;
    }
    // member function pointer
    {
    void (A_void_2_noexcept::*fp)(int) noexcept = &A_void_2_noexcept::mem1;
    std::reference_wrapper<void (A_void_2_noexcept::*)(int) noexcept> r1(fp);
    A_void_2_noexcept a;
    int i = 3;
    r1(a, i);
    assert(count == save_count+3);
    save_count = count;
    A_void_2_noexcept* ap = &a;
    r1(ap, i);
    INVOKE_NOEXCEPT(true, r1(ap, i));
    assert(count == save_count+3);
    save_count = count;
    }
    // const member function pointer
    {
    void (A_void_2_noexcept::*fp)(int) const noexcept = &A_void_2_noexcept::mem2;
    std::reference_wrapper<void (A_void_2_noexcept::*)(int) const noexcept> r1(fp);
    A_void_2_noexcept a;
    int i = 4;
    r1(a, i);
    INVOKE_NOEXCEPT(true, r1(a, i));
    assert(count == save_count+4);
    save_count = count;
    A_void_2_noexcept* ap = &a;
    r1(ap, i);
    INVOKE_NOEXCEPT(true, r1(ap, i));
    assert(count == save_count+4);
    save_count = count;
    }
}

// 2 arg, return int, noexcept

int f_int_2_noexcept(int i, int j) noexcept
{
    return i+j;
}

struct A_int_2_noexcept
{
    int operator()(int i, int j) noexcept
    {
        return i+j;
    }

    int mem1(int i) noexcept {return i+1;}
    int mem2(int i) const noexcept {return i+2;}
};

void
test_int_2_noexcept()
{
    // function
    {
    std::reference_wrapper<int (int, int) noexcept> r1(f_int_2_noexcept);
    int i = 2;
    int j = 3;
    assert(r1(i, j) == i+j);
    INVOKE_NOEXCEPT(true, r1(i, j));
    }
    // function pointer
    {
    int (*fp)(int, int) noexcept = f_int_2_noexcept;
    std::reference_wrapper<int (*)(int, int) noexcept> r1(fp);
    int i = 3;
    int j = 4;
    assert(r1(i, j) == i+j);
    INVOKE_NOEXCEPT(true, r1(i, j));
    }
    // functor
    {
    A_int_2_noexcept a0;
    std::reference_wrapper<A_int_2_noexcept> r1(a0);
    int i = 4;
    int j = 5;
    assert(r1(i, j) == i+j);
    INVOKE_NOEXCEPT(true, r1(i, j));
    }
    // member function pointer
    {
    int(A_int_2_noexcept::*fp)(int) noexcept = &A_int_2_noexcept::mem1;
    std::reference_wrapper<int (A_int_2_noexcept::*)(int) noexcept> r1(fp);
    A_int_2_noexcept a;
    int i = 3;
    assert(r1(a, i) == i+1);
    INVOKE_NOEXCEPT(true, r1(a, i));
    A_int_2_noexcept* ap = &a;
    assert(r1(ap, i) == i+1);
    INVOKE_NOEXCEPT(true, r1(ap, i));
    }
    // const member function pointer
    {
    int (A_int_2_noexcept::*fp)(int) const noexcept = &A_int_2_noexcept::mem2;
    std::reference_wrapper<int (A_int_2_noexcept::*)(int) const noexcept> r1(fp);
    A_int_2_noexcept a;
    int i = 4;
    assert(r1(a, i) == i+2);
    INVOKE_NOEXCEPT(true, r1(a, i));
    A_int_2_noexcept* ap = &a;
    assert(r1(ap, i) == i+2);
    INVOKE_NOEXCEPT(true, r1(ap, i));
    }
}

#endif // TEST_STD_VER >= 11

int main(int, char**)
{
    test_void_1();
    test_int_1();
    test_void_2();
    test_int_2();
#if TEST_STD_VER >= 11
    test_void_1_noexcept();
    test_int_1_noexcept();
    test_void_2_noexcept();
    test_int_2_noexcept();
#endif // TEST_STD_VER >= 11

  return 0;
}
