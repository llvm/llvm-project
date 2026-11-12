//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Code on Windows expects to be able to do:
//
//  #define _USE_MATH_DEFINES
//  #include <math.h>
//
// and receive the definitions of mathematical constants, even if <math.h>
// has previously been included. Make sure that works.
//

#ifdef _MSC_VER
#   include <math.h>
#   define _USE_MATH_DEFINES
#   include <math.h>

#   ifndef M_PI
#       error M_PI not defined
#   endif
#endif
