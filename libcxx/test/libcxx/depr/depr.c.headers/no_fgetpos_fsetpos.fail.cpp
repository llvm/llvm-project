//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support
// UNSUPPORTED: c++98 || c++03

#include <cstdio>

using U = decltype(::fgetpos);
using V = decltype(::fsetpos);
#ifdef _LIBCPP_HAS_NO_FGETPOS_FSETPOS
// expected-error@-3 {{no member named 'fgetpos' in the global namespace}}
// expected-error@-3 {{no member named 'fsetpos' in the global namespace}}
#else
// expected-no-diagnostics
#endif
