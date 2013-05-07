//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class InputIterator, class ForwardIterator>
//   ForwardIterator
//   uninitialized_copy(InputIterator first, InputIterator last,
//                      ForwardIterator result);

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
    B b[N];
    try
    {
        std::uninitialized_copy(b, b+N, bp);
        assert(false);
    }
    catch (...)
    {
        for (int i = 0; i < N; ++i)
            assert(bp[i].data_ == 0);
    }
    B::count_ = 0;
    std::uninitialized_copy(b, b+2, bp);
    for (int i = 0; i < 2; ++i)
        assert(bp[i].data_ == 1);
}
