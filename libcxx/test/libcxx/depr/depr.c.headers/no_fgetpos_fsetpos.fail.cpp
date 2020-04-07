//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

#include <cstdio>

using T = decltype(::fgetpos);
#ifdef _LIBCPP_HAS_NO_FGETPOS_FSETPOS
// expected-error@-2 {{no such thing}}
#else
// expected-no-diagnostics
#endif

using U = decltype(::fsetpos);
#ifdef _LIBCPP_HAS_NO_FGETPOS_FSETPOS
// expected-error@-2 {{no such thing}}
#else
// expected-no-diagnostics
#endif
