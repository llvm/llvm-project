//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <any>

// any() noexcept;

#include <any>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"
#include "count_new.hpp"

#if TEST_HAS_BUILTIN_IDENTIFIER(__has_constant_initializer)
// std::any must have a constexpr default constructor, but it's a non-literal
// type so we can't create a constexpr variable. This tests that we actually
// get 'constant initialization'.
std::any a;
static_assert(__has_constant_initializer(a),
              "any must be constant initializable");
#endif

int main()
{
    using std::any;
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
