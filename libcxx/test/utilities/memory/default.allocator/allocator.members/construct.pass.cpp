//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// template <class... Args> void construct(pointer p, Args&&... args);

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

int new_called = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    ++new_called;
    assert(s == 3 * sizeof(int));
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    --new_called;
    std::free(p);
}

int A_constructed = 0;

struct A
{
    int data;
    A() {++A_constructed;}

    A(const A&) {++A_constructed;}

    explicit A(int) {++A_constructed;}
    A(int, int*) {++A_constructed;}

    ~A() {--A_constructed;}
};

int move_only_constructed = 0;

class move_only
{
    int data;
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    move_only(const move_only&);
    move_only& operator=(const move_only&);
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    move_only(move_only&);
    move_only& operator=(move_only&);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

public:

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    move_only(move_only&&) {++move_only_constructed;}
    move_only& operator=(move_only&&) {return *this;}
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    operator std::__rv<move_only> () {return std::__rv<move_only>(*this);}
    move_only(std::__rv<move_only>) {++move_only_constructed;}
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

    move_only() {++move_only_constructed;}
    ~move_only() {--move_only_constructed;}
};

int main()
{
    {
    std::allocator<A> a;
    assert(new_called == 0);
    assert(A_constructed == 0);

    A* ap = a.allocate(3);
    assert(new_called == 1);
    assert(A_constructed == 0);

    a.construct(ap);
    assert(new_called == 1);
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(new_called == 1);
    assert(A_constructed == 0);

    a.construct(ap, A());
    assert(new_called == 1);
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(new_called == 1);
    assert(A_constructed == 0);

    a.construct(ap, 5);
    assert(new_called == 1);
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(new_called == 1);
    assert(A_constructed == 0);

    a.construct(ap, 5, (int*)0);
    assert(new_called == 1);
    assert(A_constructed == 1);

    a.destroy(ap);
    assert(new_called == 1);
    assert(A_constructed == 0);

    a.deallocate(ap, 3);
    assert(new_called == 0);
    assert(A_constructed == 0);
    }
    {
    std::allocator<move_only> a;
    assert(new_called == 0);
    assert(move_only_constructed == 0);

    move_only* ap = a.allocate(3);
    assert(new_called == 1);
    assert(move_only_constructed == 0);

    a.construct(ap);
    assert(new_called == 1);
    assert(move_only_constructed == 1);

    a.destroy(ap);
    assert(new_called == 1);
    assert(move_only_constructed == 0);

    a.construct(ap, move_only());
    assert(new_called == 1);
    assert(move_only_constructed == 1);

    a.destroy(ap);
    assert(new_called == 1);
    assert(move_only_constructed == 0);

    a.deallocate(ap, 3);
    assert(new_called == 0);
    assert(move_only_constructed == 0);
    }
}
