//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Check that user-specializations are diagnosed
// See [format.arg]/2

#include <format>

#if !__has_warning("-Winvalid-specialization")
// expected-no-diagnostics
#else
struct S {};

template <>
class std::basic_format_arg<S>; // expected-error {{cannot be specialized}}
#endif
