/* file: lgammaf.c */


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
// 02/22/02: Removed lgamma/gamma
//
/*
//   FUNCTIONS:    float   lgammaf(float x)
//                 float   gammaf(float x)
//                 Natural logarithm of GAMMA function
*/

#include "libm_support.h"

#include <math.h>
#include <math_private.h>

#include <lgamma-compat.h>

extern float  __libm_lgammaf(float /*x*/, int* /*signgam*/, int /*signgamsz*/);

#if BUILD_LGAMMA
float LGFUNC (lgammaf) (float x)
{
    return CALL_LGAMMA (float, __libm_lgammaf, x);
}
# if USE_AS_COMPAT
compat_symbol (libm, __lgammaf_compat, lgammaf, LGAMMA_OLD_VER);
# else
versioned_symbol (libm, __ieee754_lgammaf, lgammaf, LGAMMA_NEW_VER);
libm_alias_float_other (__ieee754_lgamma, lgamma)
# endif
# if GAMMA_ALIAS
strong_alias (LGFUNC (lgammaf), __ieee754_gammaf)
weak_alias (__ieee754_gammaf, gammaf)
# endif
#endif
