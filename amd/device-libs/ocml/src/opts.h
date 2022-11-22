/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

#define HAVE_FAST_FMA32() (__oclc_ISA_version == 7001 || __oclc_ISA_version == 8001 || __oclc_ISA_version >= 9000)
#define FINITE_ONLY_OPT() __oclc_finite_only_opt
#define UNSAFE_MATH_OPT() __oclc_unsafe_math_opt
#define DAZ_OPT() __oclc_daz_opt
#define CORRECTLY_ROUNDED_SQRT32() __oclc_correctly_rounded_sqrt32

// GFX6 had unhelpful handling of infinities in v_frexp_*
// instructions.
//
// TODO: Really there should be a generic frexp intrinsic and the
// backend could handle the hardware workarounds.
#define HAVE_BUGGY_FREXP_INSTRUCTIONS() (__oclc_ISA_version < 7000)
