//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class packaged_task<R(ArgTypes...)>

// template <class F, class Allocator>
//     explicit packaged_task(allocator_arg_t, const Allocator& a, F&& f);

#include <future>
#include <cassert>

#include "../../test_allocator.h"

class A
{
    long data_;

public:
    static int n_moves;
    static int n_copies;

    explicit A(long i) : data_(i) {}
    A(A&& a) : data_(a.data_) {++n_moves; a.data_ = -1;}
    A(const A& a) : data_(a.data_) {++n_copies;}

    long operator()(long i, long j) const {return data_ + i + j;}
};

int A::n_moves = 0;
int A::n_copies = 0;

int main()
{
    {
        std::packaged_task<double(int, char)> p(std::allocator_arg,
                                                test_allocator<A>(), A(5));
        assert(test_alloc_base::count > 0);
        assert(p.valid());
        std::future<double> f = p.get_future();
        p(3, 'a');
        assert(f.get() == 105.0);
        assert(A::n_copies == 0);
        assert(A::n_moves > 0);
    }
    assert(test_alloc_base::count == 0);
    A::n_copies = 0;
    A::n_copies = 0;
    {
        A a(5);
        std::packaged_task<double(int, char)> p(std::allocator_arg,
                                                test_allocator<A>(), a);
        assert(test_alloc_base::count > 0);
        assert(p.valid());
        std::future<double> f = p.get_future();
        p(3, 'a');
        assert(f.get() == 105.0);
        assert(A::n_copies > 0);
        assert(A::n_moves > 0);
    }
    assert(test_alloc_base::count == 0);
}
