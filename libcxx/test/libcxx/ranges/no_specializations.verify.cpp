//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Check that user-specializations are diagnosed
// See [range.adaptor.object]/5

#include <ranges>

#include "test_macros.h"

#if !__has_warning("-Winvalid-specialization") || TEST_STD_VER <= 20
// expected-no-diagnostics
#else
struct S {};

template <>
class std::ranges::range_adaptor_closure<S>; // expected-error {{cannot be specialized}}
#endif
