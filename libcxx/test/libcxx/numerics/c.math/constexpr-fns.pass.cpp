//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the overloads of std::__libcpp_{isnan,isinf,isfinite} that take
// floating-point values are evaluatable from constexpr contexts.
//
// These functions need to be constexpr in order to be called from CUDA, see
// https://reviews.llvm.org/D25403.  They don't actually need to be
// constexpr-evaluatable, but that's what we check here, since we can't check
// true constexpr-ness.
//
// UNSUPPORTED: c++03

#include <cmath>

#include "test_macros.h"

static_assert(std::__constexpr_isnan(0.) == false, "");
static_assert(std::__constexpr_isinf(0.0) == false, "");
static_assert(std::__constexpr_isfinite(0.0) == true, "");

int main(int, char**)
{

  return 0;
}
