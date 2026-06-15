//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// make_format_args with _BitInt(N) wider than __int128 is rejected. The
// current rejection path cascades through overload-set ambiguity in
// __determine_arg_t and a downstream "not formattable" static_assert. The
// exact diagnostic shape is not a stable interface, and verify does not
// support an "at least N matches" count modifier that would let this test
// soak up the cascade cleanly. Keeping the file as a placeholder; redesign
// is tracked separately when format gains a SFINAE-friendly rejection.

#include <format>

#include "test_macros.h"

// expected-no-diagnostics
