//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// template <class T>
//     T
//     atomic_exchange_explicit(volatile atomic<T>* obj, T desr, memory_order m);
// 
// template <class T>
//     T
//     atomic_exchange_explicit(atomic<T>* obj, T desr, memory_order m);

#include <atomic>
#include <cassert>

template <class T>
void
test()
{
    typedef std::atomic<T> A;
    A t;
    std::atomic_init(&t, T(1));
    assert(std::atomic_exchange_explicit(&t, T(2), std::memory_order_seq_cst)
           == T(1));
    assert(t == T(2));
    volatile A vt;
    std::atomic_init(&vt, T(3));
    assert(std::atomic_exchange_explicit(&vt, T(4), std::memory_order_seq_cst)
           == T(3));
    assert(vt == T(4));
}

struct A
{
    int i;

    explicit A(int d = 0) : i(d) {}
    A(const A& a) : i(a.i) {}
    A(const volatile A& a) : i(a.i) {}

    void operator=(const volatile A& a) volatile {i = a.i;}

    friend bool operator==(const A& x, const A& y)
        {return x.i == y.i;}
};

int main()
{
    test<A>();
    test<char>();
    test<signed char>();
    test<unsigned char>();
    test<short>();
    test<unsigned short>();
    test<int>();
    test<unsigned int>();
    test<long>();
    test<unsigned long>();
    test<long long>();
    test<unsigned long long>();
    test<wchar_t>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t>();
    test<char32_t>();
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<int*>();
    test<const int*>();
}
