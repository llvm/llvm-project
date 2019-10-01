// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// long double __floatdixf(di_int a);

#ifdef __x86_64__

#include "../int_lib.h"

long double __floatdixf(int64_t a) { return (long double)a; }

#endif // __i386__
