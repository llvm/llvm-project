//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-WINDOWS-FIXME

// type_traits

// aligned_storage
//
//  Issue 3034 added:
//  The member typedef type shall be a trivial standard-layout type.

#include <type_traits>
#include <cstddef>       // for std::max_align_t
#include "test_macros.h"

// The following tests assume naturally aligned types exist
// up to 64bit (double). For larger types, max_align_t should
// give the correct alignment. For pre-C++11 testing, only
// the lower bound is checked.

#if TEST_STD_VER < 11
struct natural_alignment {
    long t1;
    long long t2;
    double t3;
    long double t4;
};
#endif

int main(int, char**)
{
    {
    typedef std::aligned_storage<10, 1 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<10, 1>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef std::aligned_storage<10, 2 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<10, 2>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
    }
    {
    typedef std::aligned_storage<10, 4 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<10, 4>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 12, "");
    }
    {
    typedef std::aligned_storage<10, 8 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<10, 8>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<10, 16 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<10, 16>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<10, 32 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<10, 32>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
    }
    {
    typedef std::aligned_storage<20, 32 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<20, 32>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
    }
    {
    typedef std::aligned_storage<40, 32 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<40, 32>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 64, "");
    }
    {
    typedef std::aligned_storage<12, 16 >::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<12, 16>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<1>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<1>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 1, "");
    }
    {
    typedef std::aligned_storage<2>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<2>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 2, "");
    }
    {
    typedef std::aligned_storage<3>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<3>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef std::aligned_storage<4>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<4>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
    }
    {
    typedef std::aligned_storage<5>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<5>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef std::aligned_storage<7>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<7>);
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef std::aligned_storage<8>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<8>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 8, "");
    }
    {
    typedef std::aligned_storage<9>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<9>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<15>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<15>);
#endif
#if TEST_STD_VER <= 17
    static_assert(std::is_pod<T1>::value, "");
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<16>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<16>);
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
#if TEST_STD_VER >= 11
    const size_t alignment = TEST_ALIGNOF(std::max_align_t) > 16 ?
        16 : TEST_ALIGNOF(std::max_align_t);
    static_assert(std::alignment_of<T1>::value == alignment, "");
#else
    static_assert(std::alignment_of<T1>::value >=
                  TEST_ALIGNOF(natural_alignment), "");
    static_assert(std::alignment_of<T1>::value <= 16, "");
#endif
    static_assert(sizeof(T1) == 16, "");
    }
    {
    typedef std::aligned_storage<17>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<17>);
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
#if TEST_STD_VER >= 11
    const size_t alignment = TEST_ALIGNOF(std::max_align_t) > 16 ?
        16 : TEST_ALIGNOF(std::max_align_t);
    static_assert(std::alignment_of<T1>::value == alignment, "");
    static_assert(sizeof(T1) == 16 + alignment, "");
#else
    static_assert(std::alignment_of<T1>::value >=
                  TEST_ALIGNOF(natural_alignment), "");
    static_assert(std::alignment_of<T1>::value <= 16, "");
    static_assert(sizeof(T1) % TEST_ALIGNOF(natural_alignment) == 0, "");
#endif
    }
    {
    typedef std::aligned_storage<10>::type T1;
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(T1, std::aligned_storage_t<10>);
#endif
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
    }
  {
    const int Align = 65536;
    typedef typename std::aligned_storage<1, Align>::type T1;
    static_assert(std::is_trivial<T1>::value, "");
    static_assert(std::is_standard_layout<T1>::value, "");
    static_assert(std::alignment_of<T1>::value == Align, "");
    static_assert(sizeof(T1) == Align, "");
  }

  return 0;
}
