//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads

// <hazard_pointer>

// Check that user-specializations of hazard_pointer_obj_base are diagnosed. A program-defined
// specialization cannot meet the requirements of the original template ([namespace.std]/2): retire()
// must hand the object to the library's hazard pointer domain.

#include <hazard_pointer>

#include "test_macros.h"

#if !__has_warning("-Winvalid-specialization")
// expected-no-diagnostics
#else
struct S {};

template <>
class std::hazard_pointer_obj_base<S>; // expected-error {{cannot be specialized}}
#endif
