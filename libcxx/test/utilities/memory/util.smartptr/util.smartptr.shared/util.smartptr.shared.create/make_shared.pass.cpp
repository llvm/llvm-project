//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T, class... Args> shared_ptr<T> make_shared(Args&&... args);

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

int new_count = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    ++new_count;
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    std::free(p);
}

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
private:
    int int_;
    char char_;
};

int A::count = 0;

int main()
{
    int nc = new_count;
    {
    int i = 67;
    char c = 'e';
    std::shared_ptr<A> p = std::make_shared<A>(i, c);
    assert(new_count == nc+1);
    assert(A::count == 1);
    assert(p->get_int() == 67);
    assert(p->get_char() == 'e');
    }
    assert(A::count == 0);
}
