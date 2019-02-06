//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// unary_negate

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::unary_negate<std::logical_not<int> > F;
    const F f = F(std::logical_not<int>());
    static_assert((std::is_same<F::argument_type, int>::value), "" );
    static_assert((std::is_same<F::result_type, bool>::value), "" );
    assert(f(36));
    assert(!f(0));
}
