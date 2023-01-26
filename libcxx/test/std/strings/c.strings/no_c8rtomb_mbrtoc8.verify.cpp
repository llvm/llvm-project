//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// The system-provided <uchar.h> seems to be broken on AIX:
//   error: 'error' diagnostics seen but not expected:
//     File /usr/include/uchar.h Line 38: cannot combine with previous 'type-name' declaration specifier
//     File /usr/include/uchar.h Line 39: cannot combine with previous 'type-name' declaration specifier
//   2 errors generated.
// XFAIL: LIBCXX-AIX-FIXME

#include <uchar.h>

using U = decltype(::c8rtomb);
using V = decltype(::mbrtoc8);
#if defined(_LIBCPP_HAS_NO_C8RTOMB_MBRTOC8)
// expected-error@-3 {{no member named 'c8rtomb' in the global namespace}}
// expected-error@-3 {{no member named 'mbrtoc8' in the global namespace}}
#else
// expected-no-diagnostics
#endif
