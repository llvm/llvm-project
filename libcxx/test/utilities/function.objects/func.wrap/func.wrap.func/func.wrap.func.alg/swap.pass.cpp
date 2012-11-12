//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// template <MoveConstructible  R, MoveConstructible ... ArgTypes>
//   void swap(function<R(ArgTypes...)>&, function<R(ArgTypes...)>&);

#include <functional>
#include <new>
#include <cstdlib>
#include <cassert>

int new_called = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    ++new_called;
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    --new_called;
    std::free(p);
}

class A
{
    int data_[10];
public:
    static int count;

    explicit A(int j)
    {
        ++count;
        data_[0] = j;
    }

    A(const A& a)
    {
        ++count;
        for (int i = 0; i < 10; ++i)
            data_[i] = a.data_[i];
    }

    ~A() {--count;}

    int operator()(int i) const
    {
        for (int j = 0; j < 10; ++j)
            i += data_[j];
        return i;
    }

    int id() const {return data_[0];}
};

int A::count = 0;

int g(int) {return 0;}
int h(int) {return 1;}

int main()
{
    assert(new_called == 0);
    {
    std::function<int(int)> f1 = A(1);
    std::function<int(int)> f2 = A(2);
    assert(A::count == 2);
    assert(new_called == 2);
    assert(f1.target<A>()->id() == 1);
    assert(f2.target<A>()->id() == 2);
    swap(f1, f2);
    assert(A::count == 2);
    assert(new_called == 2);
    assert(f1.target<A>()->id() == 2);
    assert(f2.target<A>()->id() == 1);
    }
    assert(A::count == 0);
    assert(new_called == 0);
    {
    std::function<int(int)> f1 = A(1);
    std::function<int(int)> f2 = g;
    assert(A::count == 1);
    assert(new_called == 1);
    assert(f1.target<A>()->id() == 1);
    assert(*f2.target<int(*)(int)>() == g);
    swap(f1, f2);
    assert(A::count == 1);
    assert(new_called == 1);
    assert(*f1.target<int(*)(int)>() == g);
    assert(f2.target<A>()->id() == 1);
    }
    assert(A::count == 0);
    assert(new_called == 0);
    {
    std::function<int(int)> f1 = g;
    std::function<int(int)> f2 = A(1);
    assert(A::count == 1);
    assert(new_called == 1);
    assert(*f1.target<int(*)(int)>() == g);
    assert(f2.target<A>()->id() == 1);
    swap(f1, f2);
    assert(A::count == 1);
    assert(new_called == 1);
    assert(f1.target<A>()->id() == 1);
    assert(*f2.target<int(*)(int)>() == g);
    }
    assert(A::count == 0);
    assert(new_called == 0);
    {
    std::function<int(int)> f1 = g;
    std::function<int(int)> f2 = h;
    assert(A::count == 0);
    assert(new_called == 0);
    assert(*f1.target<int(*)(int)>() == g);
    assert(*f2.target<int(*)(int)>() == h);
    swap(f1, f2);
    assert(A::count == 0);
    assert(new_called == 0);
    assert(*f1.target<int(*)(int)>() == h);
    assert(*f2.target<int(*)(int)>() == g);
    }
    assert(A::count == 0);
    assert(new_called == 0);
}
