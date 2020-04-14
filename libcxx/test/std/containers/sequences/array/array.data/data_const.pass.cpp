//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// const T* data() const;

#include <array>
#include <cassert>
#include <cstddef>       // for std::max_align_t

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

struct NoDefault {
  NoDefault(int) {}
};

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
        typedef double T;
        typedef std::array<T, 3> C;
        const C c = {1, 2, 3.5};
        const T* p = c.data();
        assert(p[0] == 1);
        assert(p[1] == 2);
        assert(p[2] == 3.5);
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        const C c = {};
        const T* p = c.data();
        (void)p; // to placate scan-build
    }
    {
      typedef NoDefault T;
      typedef std::array<T, 0> C;
      const C c = {};
      const T* p = c.data();
      LIBCPP_ASSERT(p != nullptr);
    }
    {
#if TEST_STD_VER < 11
      typedef natural_alignment T;
#else
      typedef std::max_align_t T;
#endif
      typedef std::array<T, 0> C;
      const C c = {};
      const T* p = c.data();
      LIBCPP_ASSERT(p != nullptr);
      std::uintptr_t pint = reinterpret_cast<std::uintptr_t>(p);
      assert(pint % TEST_ALIGNOF(T) == 0);
    }
#if TEST_STD_VER > 14
    {
        typedef std::array<int, 5> C;
        constexpr C c1{0,1,2,3,4};
        constexpr const C c2{0,1,2,3,4};

        static_assert (  c1.data()  == &c1[0], "");
        static_assert ( *c1.data()  ==  c1[0], "");
        static_assert (  c2.data()  == &c2[0], "");
        static_assert ( *c2.data()  ==  c2[0], "");
    }
#endif

  return 0;
}
