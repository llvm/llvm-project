//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class lognormal_distribution

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);
//   https://llvm.org/PR52906

#include <random>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::lognormal_distribution<> D;
    typedef D::param_type P;
    typedef std::mt19937 G;
    G g;
    D d;

    const P p1 = d.param();
    const P p2 = d.param();
    assert(p1 == p2);
    (void) d(g, p1); // This line must not modify p1.
    assert(p1 == p2);
    LIBCPP_ASSERT(p1 == d.param());

    return 0;
}
