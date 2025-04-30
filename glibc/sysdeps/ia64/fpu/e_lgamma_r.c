/* file: lgamma_r.c */


// Copyright (c) 2002 Intel Corporation
// All rights reserved.
//
// Contributed 2002 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//
//

// History
//==============================================================
// 02/04/02: Initial version
// 02/22/02: Removed lgammaf_r, gammaf_r
/*
//   FUNCTIONS:    double  lgamma_r(double x, int* signgam)
//                 double  gamma_r(double x, int* signgam)
//                 Natural logarithm of GAMMA function
*/

#include "libm_support.h"


extern double __libm_lgamma(double /*x*/, int* /*signgam*/, int /*signgamsz*/);


double __ieee754_lgamma_r(double x, int* signgam)
{
    return __libm_lgamma(x, signgam, sizeof(*signgam));
}
libm_alias_double_r (__ieee754_lgamma, lgamma, _r)

#ifndef _LIBC
double __ieee754_gamma_r(double x, int* signgam)
{
    return __libm_lgamma(x, signgam, sizeof(*signgam));
}
weak_alias (__ieee754_gamma_r, gamma_r)
#endif
