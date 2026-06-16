//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// Placeholder for verify coverage of make_format_args with _BitInt(N > 128).
// The current rejection cascades through unstable diagnostics; will be filled
// in once format gains a SFINAE-friendly rejection path.

#include <format>

#include "test_macros.h"

// expected-no-diagnostics
