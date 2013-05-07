//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// aligned_storage

#include <type_traits>

int main()
{
    {
    typedef std::aligned_storage<10, 1 >::type T1;
    static_assert(std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef std::aligned_storage<10, 2 >::type T1;
    static_assert(std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef std::aligned_storage<10, 4 >::type T1;
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 12, "");
    }
    {
    typedef std::aligned_storage<10, 8 >::type T1;
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<10, 16 >::type T1;
    static_assert(std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<10, 32 >::type T1;
    static_assert(std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
    }
    {
    typedef std::aligned_storage<20, 32 >::type T1;
    static_assert(std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
    }
    {
    typedef std::aligned_storage<40, 32 >::type T1;
    static_assert(std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 64, "");
    }
    {
    typedef std::aligned_storage<12, 16 >::type T1;
    static_assert(std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<1>::type T1;
    static_assert(std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 1, "");
    }
    {
    typedef std::aligned_storage<2>::type T1;
    static_assert(std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 2, "");
    }
    {
    typedef std::aligned_storage<3>::type T1;
    static_assert(std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef std::aligned_storage<4>::type T1;
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef std::aligned_storage<5>::type T1;
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef std::aligned_storage<7>::type T1;
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef std::aligned_storage<8>::type T1;
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef std::aligned_storage<9>::type T1;
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<15>::type T1;
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<16>::type T1;
    static_assert(std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<17>::type T1;
    static_assert(std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 32, "");
    }
    {
    typedef std::aligned_storage<10>::type T1;
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
}
