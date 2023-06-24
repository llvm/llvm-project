/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// OCML prototypes
#include "ocml.h"

// Tables
#include "tables.h"

// Builtins
#include "builtins.h"

// Mangling
#define MATH_MANGLE(N) OCML_MANGLE_F32(N)
#define MATH_MANGLE2(N) OCML_MANGLE_2F32(N)
#define MATH_PRIVATE(N) MANGLE3(__ocmlpriv,N,f32)

// Optimization Controls
#include "opts.h"

// Attributes
#define PUREATTR __attribute__((pure))
#define CONSTATTR __attribute__((const))

// Math controls
#include "privF.h"

// Floating point patterns
#define SIGNBIT_SP32      (int)0x80000000
#define EXSIGNBIT_SP32    0x7fffffff
#define EXPBITS_SP32      0x7f800000
#define MANTBITS_SP32     0x007fffff
#define ONEEXPBITS_SP32   0x3f800000
#define TWOEXPBITS_SP32   0x40000000
#define HALFEXPBITS_SP32  0x3f000000
#define IMPBIT_SP32       0x00800000
#define QNANBITPATT_SP32  0x7fc00000
#define PINFBITPATT_SP32  0x7f800000
#define NINFBITPATT_SP32  (int)0xff800000
#define EXPBIAS_SP32      127
#define EXPSHIFTBITS_SP32 23
#define BIASEDEMIN_SP32   1
#define EMIN_SP32         -126
#define BIASEDEMAX_SP32   254
#define EMAX_SP32         127
#define MANTLENGTH_SP32   24
#define BASEDIGITS_SP32   7

#define QNAN_F32 __builtin_nanf("")
#define PINF_F32 __builtin_inff()
#define NINF_F32 (-__builtin_inff())
