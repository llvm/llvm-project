//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %compile -fsyntax-only

#ifdef _MSC_VER

#include <math.h>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#error M_PI not defined
#endif

#endif
