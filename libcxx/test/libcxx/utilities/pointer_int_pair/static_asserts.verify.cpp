//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include <__utility/pointer_int_pair.h>
#include <cstddef>

#include "test_macros.h"

// expected-error@*:* {{Not enough bits available for requested bit count}}
std::__pointer_int_pair<char*, size_t, std::__integer_width{2}> ptr1; // expected-note {{here}}
// expected-error@*:* {{_IntType has to be an integral type}}
std::__pointer_int_pair<int*, int*, std::__integer_width{2}> ptr2; // expected-note {{here}}
// expected-error@*:* {{__pointer_int_pair doesn't work for signed types}}
std::__pointer_int_pair<int*, signed, std::__integer_width{2}> ptr3; // expected-note {{here}}
