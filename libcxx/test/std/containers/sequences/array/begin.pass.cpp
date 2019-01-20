//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// iterator begin();

#include <array>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

struct NoDefault {
  NoDefault(int) {}
};


int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};
        C::iterator i;
        i = c.begin();
        assert(*i == 1);
        assert(&*i == c.data());
        *i = 5.5;
        assert(c[0] == 5.5);
    }
    {
      typedef NoDefault T;
      typedef std::array<T, 0> C;
      C c = {};
      C::iterator ib, ie;
      ib = c.begin();
      ie = c.end();
      assert(ib == ie);
      LIBCPP_ASSERT(ib != nullptr);
      LIBCPP_ASSERT(ie != nullptr);
    }
}
