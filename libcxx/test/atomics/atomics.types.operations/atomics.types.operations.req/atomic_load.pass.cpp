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
//     atomic_load(const volatile atomic<T>* obj);
// 
// template <class T>
//     T
//     atomic_load(const atomic<T>* obj);

#include <atomic>
#include <type_traits>
#include <cassert>

template <class T>
void
test()
{
    typedef std::atomic<T> A;
    A t;
    std::atomic_init(&t, T(1));
    assert(std::atomic_load(&t) == T(1));
    volatile A vt;
    std::atomic_init(&vt, T(2));
    assert(std::atomic_load(&vt) == T(2));
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
