//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <uchar.h>

#include "test_macros.h"

// When C++ char8_t support is not enabled, definitions of these functions that
// match the C2X declarations may still be present in the global namespace with
// a char8_t typedef substituted for the C++ char8_t type. If so, these are not
// the declarations we are looking for, so don't test for them.
#if !defined(TEST_HAS_NO_CHAR8_T)
using U = decltype(::c8rtomb);
using V = decltype(::mbrtoc8);
#  if defined(_LIBCPP_HAS_NO_C8RTOMB_MBRTOC8)
// expected-error@-3 {{no member named 'c8rtomb' in the global namespace}}
// expected-error@-3 {{no member named 'mbrtoc8' in the global namespace}}
#  else
// expected-no-diagnostics
#  endif
#else
// expected-no-diagnostics
#endif
