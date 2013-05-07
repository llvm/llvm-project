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
//     void
//     atomic_store(volatile atomic<T>* obj, T desr);
// 
// template <class T>
//     void
//     atomic_store(atomic<T>* obj, T desr);

#include <atomic>
#include <type_traits>
#include <cassert>

template <class T>
void
test()
{
    typedef std::atomic<T> A;
    A t;
    std::atomic_store(&t, T(1));
    assert(t == T(1));
    volatile A vt;
    std::atomic_store(&vt, T(2));
    assert(vt == T(2));
}

struct A
{
    int i;

    explicit A(int d = 0) : i(d) {}

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
