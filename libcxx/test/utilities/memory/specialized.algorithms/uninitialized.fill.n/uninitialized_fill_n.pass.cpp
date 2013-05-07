//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class ForwardIterator, class Size, class T>
//   ForwardIterator
//   uninitialized_fill_n(ForwardIterator first, Size n, const T& x);

#include <memory>
#include <cassert>

struct B
{
    static int count_;
    int data_;
    explicit B() : data_(1) {}
    B(const B& b) {if (++count_ == 3) throw 1; data_ = b.data_;}
    ~B() {data_ = 0;}
};

int B::count_ = 0;

int main()
{
    const int N = 5;
    char pool[sizeof(B)*N] = {0};
    B* bp = (B*)pool;
    try
    {
        std::uninitialized_fill_n(bp, 5, B());
        assert(false);
    }
    catch (...)
    {
        for (int i = 0; i < N; ++i)
            assert(bp[i].data_ == 0);
    }
    B::count_ = 0;
    B* r = std::uninitialized_fill_n(bp, 2, B());
    assert(r == bp + 2);
    for (int i = 0; i < 2; ++i)
        assert(bp[i].data_ == 1);
}
