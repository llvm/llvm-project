//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/any>

// any() noexcept;

#include <experimental/any>
#include <type_traits>
#include <cassert>

#include "experimental_any_helpers.h"
#include "count_new.hpp"


int main()
{
    using std::experimental::any;
    {
        static_assert(
            std::is_nothrow_default_constructible<any>::value
          , "Must be default constructible"
          );
    }
    {
        DisableAllocationGuard g; ((void)g);
        any const a;
        assertEmpty(a);
    }
}
