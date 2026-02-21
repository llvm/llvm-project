//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Check that user-specializations are diagnosed
// See [indirect.general]/8

#include <memory>

#if !__has_warning("-Winvalid-specialization")
// expected-no-diagnostics
#else
struct S {};

template <>
class std::indirect<S>; // expected-error {{cannot be specialized}}
#endif
